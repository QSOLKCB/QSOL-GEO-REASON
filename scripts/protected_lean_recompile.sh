#!/usr/bin/env bash
set -Eeu -o pipefail
trap 'status=$?; echo "protected-recompile failure status=${status} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "$status"' ERR

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${LEAN_HOME:?LEAN_HOME is required}"
: "${PINNED_LEAN_BIN_SHA256:?PINNED_LEAN_BIN_SHA256 is required}"
: "${PINNED_AUDIT_SHA256:?PINNED_AUDIT_SHA256 is required}"

audit_root="/opt/qsol-lean-audit"
protected_base="$audit_root/recompiled"
dependency_path_file="$audit_root/dependency-lean-path"
audit_path_file="$audit_root/lean-path"
source_receipt="$audit_root/project-source-receipt.json"
project_build_lib="$GITHUB_WORKSPACE/.lake/build/lib/lean"
source_root="$GITHUB_WORKSPACE/Lean"
source_purity_script="$GITHUB_WORKSPACE/scripts/verify_lean_source_purity.py"

lean_bin="$LEAN_HOME/bin/lean"
test -x "$lean_bin"
test -f "$source_purity_script"
test ! -L "$source_purity_script"
test "$(sha256sum "$lean_bin" | awk '{print $1}')" = "$PINNED_LEAN_BIN_SHA256"
test "$(sha256sum "$audit_root/Audit.lean" | awk '{print $1}')" = "$PINNED_AUDIT_SHA256"
source_purity_sha="$(sha256sum "$source_purity_script" | awk '{print $1}')"

# Establish a root-owned receipt for the complete, closed production source
# surface before any project module is elaborated. The verifier rejects all
# project-defined compile-time execution mechanisms, including run_cmd,
# run_tac, initializers, custom elaborators/macros, unsafe declarations,
# foreign hooks, native evaluation, and IO/process/filesystem APIs.
tmp_source_receipt="$(mktemp)"
/usr/bin/python3 "$source_purity_script" \
  --root "$source_root" \
  --self-test \
  --receipt "$tmp_source_receipt" \
  --write-receipt
sudo install -o root -g root -m 0444 "$tmp_source_receipt" "$source_receipt"
rm -f "$tmp_source_receipt"
test ! -L "$source_receipt"

verify_project_source() {
  test "$(sha256sum "$source_purity_script" | awk '{print $1}')" = "$source_purity_sha"
  /usr/bin/python3 "$source_purity_script" \
    --root "$source_root" \
    --receipt "$source_receipt"
}

terminate_qsolcompile() {
  sudo pkill -TERM -u qsolcompile 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! sudo pgrep -u qsolcompile >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  sudo pkill -KILL -u qsolcompile 2>/dev/null || true
  sleep 1
  if sudo pgrep -u qsolcompile >/dev/null 2>&1; then
    echo "qsolcompile descendant survived protected compilation boundary" >&2
    exit 1
  fi
}

# The compiler identity exists only for direct, source-bound project
# recompilation. It never evaluates the project lakefile and owns no reviewed
# source, dependency artifact, protected audit input, or prior module root.
sudo useradd --system --no-create-home --shell /usr/sbin/nologin qsolcompile 2>/dev/null || true
terminate_qsolcompile
sudo rm -rf /tmp/qsolcompile-home
sudo install -d -o qsolcompile -g qsolcompile -m 0700 /tmp/qsolcompile-home

if sudo -u qsolcompile test -w "$source_purity_script"; then
  echo "Protected compiler can write source-purity verifier" >&2
  exit 1
fi
if sudo -u qsolcompile test -w "$source_receipt"; then
  echo "Protected compiler can write production source receipt" >&2
  exit 1
fi

# Build a dependency-only search path. In particular, the project output made
# by qsolbuild is deliberately excluded from the protected theorem audit.
tmp_dependency_path="$(mktemp)"
: > "$tmp_dependency_path"
dependency_libs=0
for pkg in "$GITHUB_WORKSPACE"/.lake/packages/*; do
  [ -d "$pkg" ] || continue
  test ! -L "$pkg"
  lib="$pkg/.lake/build/lib/lean"
  [ -d "$lib" ] || continue
  test ! -L "$pkg/.lake"
  test ! -L "$pkg/.lake/build"
  test ! -L "$pkg/.lake/build/lib"
  test ! -L "$lib"
  if [ "$dependency_libs" -gt 0 ]; then
    printf ':' >> "$tmp_dependency_path"
  fi
  printf '%s' "$lib" >> "$tmp_dependency_path"
  dependency_libs="$((dependency_libs + 1))"
done
test "$dependency_libs" -gt 0
printf '\n' >> "$tmp_dependency_path"
sudo install -o root -g root -m 0444 "$tmp_dependency_path" "$dependency_path_file"
rm -f "$tmp_dependency_path"

dependency_lean_path="$(cat "$dependency_path_file")"
test -n "$dependency_lean_path"
case ":$dependency_lean_path:" in
  *":$project_build_lib:"*)
    echo "qsolbuild project library leaked into protected dependency path" >&2
    exit 1
    ;;
esac

sudo rm -rf "$protected_base"
sudo install -d -o root -g root -m 0555 "$protected_base"

# Lean resolves a top-level module package such as GeoReason from one search
# root. Keep the one-shot compiler roots for isolation/provenance, but publish
# each frozen object into this single root-owned package tree for subsequent
# imports and the final audit.
assembled_root="$protected_base/assembled"
sudo install -d -o root -g root -m 0555 "$assembled_root"
test ! -L "$assembled_root"

compile_protected_module() {
  label="$1"
  source_file="$2"
  output_rel="$3"
  lean_path="$4"
  module_root="$protected_base/$label"
  output_file="$module_root/$output_rel"
  output_dir="$(dirname "$output_file")"

  verify_project_source
  test -f "$source_file"
  test ! -L "$source_file"
  if sudo -u qsolcompile test -w "$source_file"; then
    echo "Protected compiler can write reviewed source: $source_file" >&2
    exit 1
  fi
  if sudo -u qsolcompile test -w "$lean_bin"; then
    echo "Protected compiler can write pinned Lean binary" >&2
    exit 1
  fi

  # Give this compiler invocation one fresh output root and one fresh HOME.
  # Prior roots are already root-owned/read-only, so later elaboration cannot
  # replace earlier reviewed objects.
  terminate_qsolcompile
  sudo rm -rf /tmp/qsolcompile-home
  sudo install -d -o qsolcompile -g qsolcompile -m 0700 /tmp/qsolcompile-home
  sudo install -d -o qsolcompile -g qsolcompile -m 0700 "$module_root" "$output_dir"
  if sudo -u qsolbuild test -w "$module_root"; then
    echo "qsolbuild can write protected compiler output root: $module_root" >&2
    exit 1
  fi

  sudo -u qsolcompile env -i \
    HOME=/tmp/qsolcompile-home \
    PATH="$LEAN_HOME/bin:/usr/bin:/bin" \
    LEAN_PATH="$lean_path" \
    LEAN_NUM_THREADS=1 \
    "$lean_bin" \
      -R "$source_root" \
      -DwarningAsError=true \
      -o "$output_file" \
      "$source_file"

  # The production subset forbids source-controlled compile-time execution.
  # Terminate the compiler identity before examining the accepted object anyway,
  # closing every descriptor and providing defense in depth against descendants.
  terminate_qsolcompile
  verify_project_source

  # Verify the emitted object only after the compiler identity is process-free,
  # then transfer the entire root to read-only root ownership.
  sudo test -f "$output_file"
  sudo test ! -L "$output_file"
  output_sha="$(sudo sha256sum "$output_file" | awk '{print $1}')"
  sudo chown -R root:root "$module_root"
  sudo find "$module_root" -type d -exec chmod 0555 {} +
  sudo find "$module_root" -type f -exec chmod 0444 {} +
  test "$(sudo sha256sum "$output_file" | awk '{print $1}')" = "$output_sha"

  for identity in qsolbuild qsolcompile qsolaudit; do
    if sudo -u "$identity" test -w "$module_root"; then
      echo "$identity can write frozen protected module root: $module_root" >&2
      exit 1
    fi
  done

  # Publish the frozen object into the single package root Lean requires for
  # GeoReason lookup. Publication is root-only, copy-based, hash-checked, and
  # never gives qsolbuild/qsolcompile/qsolaudit a writable assembled surface.
  assembled_file="$assembled_root/$output_rel"
  assembled_dir="$(dirname "$assembled_file")"
  sudo install -d -o root -g root -m 0555 "$assembled_dir"
  sudo test ! -L "$assembled_dir"
  sudo test ! -e "$assembled_file"
  sudo install -o root -g root -m 0444 "$output_file" "$assembled_file"
  sudo test -f "$assembled_file"
  sudo test ! -L "$assembled_file"
  test "$(sudo sha256sum "$output_file" | awk '{print $1}')" = \
       "$(sudo sha256sum "$assembled_file" | awk '{print $1}')"

  for identity in qsolbuild qsolcompile qsolaudit; do
    if sudo -u "$identity" test -w "$assembled_file"; then
      echo "$identity can write assembled protected module: $assembled_file" >&2
      exit 1
    fi
  done
}

trajectory_root="$protected_base/00-trajectory"
cosine_root="$protected_base/01-cosine"
menger_root="$protected_base/02-menger"
arclength_root="$protected_base/03-arclength"
geo_root="$protected_base/04-georeason"

compile_protected_module \
  "00-trajectory" \
  "$source_root/GeoReason/Trajectory.lean" \
  "GeoReason/Trajectory.olean" \
  "$dependency_lean_path"

compile_protected_module \
  "01-cosine" \
  "$source_root/GeoReason/Cosine.lean" \
  "GeoReason/Cosine.olean" \
  "$assembled_root:$dependency_lean_path"

compile_protected_module \
  "02-menger" \
  "$source_root/GeoReason/Menger.lean" \
  "GeoReason/Menger.olean" \
  "$assembled_root:$dependency_lean_path"

compile_protected_module \
  "03-arclength" \
  "$source_root/GeoReason/ArcLength.lean" \
  "GeoReason/ArcLength.olean" \
  "$assembled_root:$dependency_lean_path"

compile_protected_module \
  "04-georeason" \
  "$source_root/GeoReason.lean" \
  "GeoReason.olean" \
  "$assembled_root:$dependency_lean_path"

terminate_qsolcompile
verify_project_source

# The final audit path contains only the assembled source-bound project graph
# plus the already authenticated dependency closure. The qsolbuild project tree
# is never eligible for import by the final protected audit.
final_lean_path="$assembled_root:$dependency_lean_path"
tmp_audit_path="$(mktemp)"
printf '%s\n' "$final_lean_path" > "$tmp_audit_path"
sudo install -o root -g root -m 0444 "$tmp_audit_path" "$audit_path_file"
rm -f "$tmp_audit_path"

if grep -Fq "$project_build_lib" "$audit_path_file"; then
  echo "Unauthenticated qsolbuild project output appears in protected LEAN_PATH" >&2
  exit 1
fi

IFS=: read -r -a audit_paths < "$audit_path_file"
for path in "${audit_paths[@]}"; do
  test -d "$path"
  test ! -L "$path"
  sudo -u qsolaudit test -r "$path"
  for identity in qsolbuild qsolcompile qsolaudit; do
    if sudo -u "$identity" test -w "$path"; then
      echo "$identity can write protected Lean module path: $path" >&2
      exit 1
    fi
  done
done

test -f "$trajectory_root/GeoReason/Trajectory.olean"
test -f "$cosine_root/GeoReason/Cosine.olean"
test -f "$menger_root/GeoReason/Menger.olean"
test -f "$arclength_root/GeoReason/ArcLength.olean"
test -f "$geo_root/GeoReason.olean"

test -f "$assembled_root/GeoReason/Trajectory.olean"
test -f "$assembled_root/GeoReason/Cosine.olean"
test -f "$assembled_root/GeoReason/Menger.olean"
test -f "$assembled_root/GeoReason/ArcLength.olean"
test -f "$assembled_root/GeoReason.olean"

test "$(sha256sum "$lean_bin" | awk '{print $1}')" = "$PINNED_LEAN_BIN_SHA256"
test "$(sha256sum "$audit_root/Audit.lean" | awk '{print $1}')" = "$PINNED_AUDIT_SHA256"
test "$(sha256sum "$source_purity_script" | awk '{print $1}')" = "$source_purity_sha"

find "$protected_base" -type f -name '*.olean' -print0 \
  | sort -z \
  | xargs -0 sha256sum

echo "Protected GeoReason recompilation accepted only a closed, non-executable production source subset and excluded qsolbuild project objects from the audit path."

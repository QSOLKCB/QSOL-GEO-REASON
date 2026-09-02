#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${LEAN_HOME:?LEAN_HOME is required}"
: "${PINNED_LEAN_BIN_SHA256:?PINNED_LEAN_BIN_SHA256 is required}"
: "${PINNED_AUDIT_SHA256:?PINNED_AUDIT_SHA256 is required}"

audit_root="/opt/qsol-lean-audit"
protected_base="$audit_root/recompiled"
dependency_path_file="$audit_root/dependency-lean-path"
audit_path_file="$audit_root/lean-path"
project_build_lib="$GITHUB_WORKSPACE/.lake/build/lib/lean"
source_root="$GITHUB_WORKSPACE/Lean"

lean_bin="$LEAN_HOME/bin/lean"
test -x "$lean_bin"
test "$(sha256sum "$lean_bin" | awk '{print $1}')" = "$PINNED_LEAN_BIN_SHA256"
test "$(sha256sum "$audit_root/Audit.lean" | awk '{print $1}')" = "$PINNED_AUDIT_SHA256"

# The compiler identity exists only for direct, source-bound project
# recompilation.  It never evaluates the project lakefile and owns no reviewed
# source, dependency artifact, protected audit input, or prior module root.
sudo useradd --system --no-create-home --shell /usr/sbin/nologin qsolcompile 2>/dev/null || true
sudo rm -rf /tmp/qsolcompile-home
sudo install -d -o qsolcompile -g qsolcompile -m 0700 /tmp/qsolcompile-home

# Build a dependency-only search path.  In particular, the project output made
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

compile_protected_module() {
  label="$1"
  source_file="$2"
  output_rel="$3"
  lean_path="$4"
  module_root="$protected_base/$label"
  output_file="$module_root/$output_rel"
  output_dir="$(dirname "$output_file")"

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

  # Give this compiler invocation one fresh output root.  Prior roots are
  # already root-owned/read-only, so later module elaboration cannot replace
  # earlier reviewed objects.
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
      "$source_file" \
      -o "$output_file"

  # A reviewed module may perform compile-time IO.  Do not allow descendants
  # of that elaboration to survive into the next module's writable root.
  sudo pkill -KILL -u qsolcompile 2>/dev/null || true

  test -f "$output_file"
  test ! -L "$output_file"
  sudo chown -R root:root "$module_root"
  sudo find "$module_root" -type d -exec chmod 0555 {} +
  sudo find "$module_root" -type f -exec chmod 0444 {} +

  for identity in qsolbuild qsolcompile qsolaudit; do
    if sudo -u "$identity" test -w "$module_root"; then
      echo "$identity can write frozen protected module root: $module_root" >&2
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
  "$trajectory_root:$dependency_lean_path"

compile_protected_module \
  "02-menger" \
  "$source_root/GeoReason/Menger.lean" \
  "GeoReason/Menger.olean" \
  "$cosine_root:$trajectory_root:$dependency_lean_path"

compile_protected_module \
  "03-arclength" \
  "$source_root/GeoReason/ArcLength.lean" \
  "GeoReason/ArcLength.olean" \
  "$menger_root:$cosine_root:$trajectory_root:$dependency_lean_path"

compile_protected_module \
  "04-georeason" \
  "$source_root/GeoReason.lean" \
  "GeoReason.olean" \
  "$arclength_root:$menger_root:$cosine_root:$trajectory_root:$dependency_lean_path"

# The audit path contains only independently recompiled project modules plus
# the already authenticated dependency closure.  The qsolbuild project tree is
# never eligible for import by the final protected audit.
final_lean_path="$geo_root:$arclength_root:$menger_root:$cosine_root:$trajectory_root:$dependency_lean_path"
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

test "$(sha256sum "$lean_bin" | awk '{print $1}')" = "$PINNED_LEAN_BIN_SHA256"
test "$(sha256sum "$audit_root/Audit.lean" | awk '{print $1}')" = "$PINNED_AUDIT_SHA256"

find "$protected_base" -type f -name '*.olean' -print0 \
  | sort -z \
  | xargs -0 sha256sum

echo "Protected GeoReason recompilation excluded qsolbuild project objects from the audit path."

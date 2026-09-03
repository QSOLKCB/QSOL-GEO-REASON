import Lake
open Lake DSL

package «qsol-geo-reason» where
  version := v!"0.1.0"

-- Protected workflows bind these exact dependency-declaration bytes to the
-- frozen manifest and verify every resolved checkout before compilation.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @
    "0df444a360eaa60ab8c11dca51a86af692955474"

@[default_target]
lean_lib GeoReason where
  srcDir := "Lean"

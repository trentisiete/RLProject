# ------------------------------------------------------
# clean_envs.ps1
#   Elimina CONDA envs (excepto base) y dirs de venv/venv/.venv
#   SOLO a nivel de usuario, no dentro de la instalación.
# ------------------------------------------------------

# 1) Eliminar entornos de Conda (salvo "base")
try {
    $envs = conda env list --json 2>$null |
            ConvertFrom-Json |
            Select-Object -ExpandProperty envs |
            ForEach-Object { Split-Path $_ -Leaf }
} catch {
    $envs = conda env list |
            Where-Object { $_ -and $_ -notmatch '^#' } |
            ForEach-Object { ($_ -split '\s+')[0] }
}

foreach ($e in $envs) {
    if ($e -and $e -ne 'base') {
        Write-Host "⏳ Eliminando Conda env: $e"
        conda env remove -n $e -y
    }
}

# 2) Eliminar solo virtualenvs creados directa en tu carpeta de usuario
$topLevelDirs = Get-ChildItem -Path $HOME -Directory -ErrorAction SilentlyContinue
$toDelete = $topLevelDirs | Where-Object { $_.Name -in @('env','venv','.venv') }

foreach ($d in $toDelete) {
    Write-Host "⏳ Borrando virtualenv folder: $($d.FullName)"
    Remove-Item $d.FullName -Recurse -Force
}

Write-Host "✅ Limpieza completada."

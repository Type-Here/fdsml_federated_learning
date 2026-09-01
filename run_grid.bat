@echo off
setlocal enabledelayedexpansion
REM ---------------------------------------------------------------------------
REM  One-command grid search on a Windows training machine.
REM  The WSL / Linux equivalent is run_grid.sh, same behaviour.
REM
REM  Clones the repository, builds a virtual environment, installs the
REM  dependencies, prepares GTSRB and runs the grid. Everything stays on this
REM  machine.
REM
REM  Safe to re-run: it skips whatever is already in place, and the grid itself
REM  deduplicates against the results CSV, so an interrupted session resumes and
REM  only the run that was in flight is lost.
REM
REM    run_grid.bat
REM    run_grid.bat grid_search_config_checkpoints.json
REM
REM  Run from inside a clone, it uses that clone and leaves git alone. Copied
REM  out and run on its own, it clones into %WORKDIR% (default %USERPROFILE%\fdsml).
REM ---------------------------------------------------------------------------

set "REPO_URL=https://github.com/Type-Here/fdsml_federated_learning.git"
set "BRANCH=features/tta"
set "TORCH_CUDA=cu126"
REM Its own environment, never the .venv used for editing on a machine without a
REM GPU: this one carries torch and would otherwise overwrite that one in place.
set "VENV_NAME=.venv-grid"

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=grid_search_config.json"

set "IN_PLACE=0"
if exist "%~dp0federated_grid_search.py" (
    if not defined WORKDIR (
        set "WORKDIR=%~dp0"
        set "IN_PLACE=1"
    )
)
if not defined WORKDIR set "WORKDIR=%USERPROFILE%\fdsml"
REM %~dp0 ends with a backslash; strip it so the paths built below stay clean.
if "%WORKDIR:~-1%"=="\" set "WORKDIR=%WORKDIR:~0,-1%"

REM --- 1. interpreter --------------------------------------------------------
REM numpy 1.26.4 and scikit-learn 1.5.0 have no wheels beyond Python 3.12.
set "PY="
for %%V in (3.11 3.12) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    echo ERROR: Python 3.11 or 3.12 is required ^(numpy 1.26.4 has no wheel beyond 3.12^).
    goto :fail
)
where git >nul 2>&1 || (echo ERROR: git not found in PATH. & goto :fail)
echo ==^> interpreter: %PY%
%PY% -V

REM --- 2. repository ---------------------------------------------------------
if "%IN_PLACE%"=="1" (
    echo ==^> running inside an existing checkout, leaving git untouched
) else (
    if exist "%WORKDIR%\.git" (
        echo ==^> repository already in %WORKDIR%, updating
        git -C "%WORKDIR%" fetch --quiet origin %BRANCH%             || goto :fail
        git -C "%WORKDIR%" checkout --quiet %BRANCH%                 || goto :fail
        git -C "%WORKDIR%" pull --quiet --ff-only origin %BRANCH%    || goto :fail
    ) else (
        echo ==^> cloning %BRANCH% into %WORKDIR%
        git clone --quiet -b %BRANCH% "%REPO_URL%" "%WORKDIR%"       || goto :fail
    )
)
cd /d "%WORKDIR%" || goto :fail
git log --oneline -1

REM --- 3. environment --------------------------------------------------------
set "VENV=%WORKDIR%\%VENV_NAME%"
set "VPY=%VENV%\Scripts\python.exe"
if not exist "%VPY%" (
    echo ==^> creating %VENV_NAME%
    %PY% -m venv "%VENV%" || goto :fail
)

if not exist "%VENV%\.deps-ok" (
    echo ==^> installing dependencies ^(a few minutes^)
    "%VPY%" -m pip install --quiet --upgrade pip            || goto :fail
    REM imagecorruptions imports pkg_resources, dropped in setuptools 81.
    "%VPY%" -m pip install --quiet "setuptools<81"          || goto :fail
    "%VPY%" -m pip install --quiet -r requirements_gpu.txt  || goto :fail

    REM torch is installed separately on purpose: the CUDA wheels do not live on
    REM PyPI, so they cannot be pinned in the requirements file.
    where nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo.
        echo WARNING: no NVIDIA GPU found. Installing the CPU build.
        echo The full grid is not practical on a CPU: use test_config.json instead.
        echo.
        "%VPY%" -m pip install --quiet torch torchvision || goto :fail
    ) else (
        echo ==^> NVIDIA GPU detected, installing torch %TORCH_CUDA%
        "%VPY%" -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/%TORCH_CUDA% || goto :fail
    )
    type nul > "%VENV%\.deps-ok"
) else (
    echo ==^> dependencies already installed ^(delete %VENV%\.deps-ok to redo them^)
)

"%VPY%" -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

REM --- 4. dataset ------------------------------------------------------------
REM The script that builds it is idempotent: it skips a download and an
REM extraction whose output is already there, so this costs nothing on a re-run.
set "NIMG=0"
if exist "dataset\gtsrb\train" (
    for /f %%C in ('dir /s /b "dataset\gtsrb\train\*.png" 2^>nul ^| find /c /v ""') do set "NIMG=%%C"
)
if not "!NIMG!"=="26640" (
    echo ==^> preparing GTSRB ^(about 200 MB to download^)
    "%VPY%" datasets_prep\prepare_gtsrb.py --splits train || goto :fail
    for /f %%C in ('dir /s /b "dataset\gtsrb\train\*.png" 2^>nul ^| find /c /v ""') do set "NIMG=%%C"
)
if not "!NIMG!"=="26640" (
    echo ERROR: expected 26640 images under dataset\gtsrb\train, found !NIMG!.
    goto :fail
)
echo ==^> dataset ready: !NIMG! images

REM --- 5. the grid -----------------------------------------------------------
if not exist "run_logs" mkdir "run_logs"
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "TS=%%T"
set "LOG=run_logs\grid_%TS%.log"
echo ==^> starting the grid with %CONFIG%
echo ==^> log: %LOG%   ^(Ctrl+C stops it; only the run in flight is lost^)
powershell -NoProfile -Command "& '%VPY%' -u federated_grid_search.py '%CONFIG%' 2>&1 | Tee-Object -FilePath '%LOG%'"

REM --- 6. what came out ------------------------------------------------------
REM Output directories are namespaced by machine name, so several PCs can share
REM a results tree without overwriting each other.
echo.
echo ==^> state
"%VPY%" -c "import csv,glob,os,socket;pc=socket.gethostname();p=os.path.join('csv_'+pc,pc,'federated_grid_search_results_'+pc+'.csv');r=list(csv.DictReader(open(p))) if os.path.exists(p) else [];print('  completed runs :',len(r));print('  results        :',p);print('  checkpoints    :',len(glob.glob(os.path.join('checkpoints_'+pc,'*.pkl'))),'.pkl files in checkpoints_'+pc)"
goto :done

:fail
echo.
echo === stopped on error ===
exit /b 1

:done
echo.
echo === done ===
endlocal

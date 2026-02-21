$ErrorActionPreference = 'Stop'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDir = 'logs_case2000_s20000_penetrations'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$masterLog = Join-Path $logDir ("master_$ts.log")
'Batch start: ' + (Get-Date) | Tee-Object -FilePath $masterLog
$log = Join-Path $logDir 'run_p05_$ts.log'
'Starting user_config_case2000_s20000_droop_strong_p05.yaml at ' + (Get-Date) | Tee-Object -FilePath $masterLog -Append
py -3.11 -m gridfm_datakit.cli generate user_config_case2000_s20000_droop_strong_p05.yaml 2>&1 | Tee-Object -FilePath $log
'Finished user_config_case2000_s20000_droop_strong_p05.yaml at ' + (Get-Date) | Tee-Object -FilePath $masterLog -Append
$log = Join-Path $logDir 'run_p10_$ts.log'
'Starting user_config_case2000_s20000_droop_strong_p10.yaml at ' + (Get-Date) | Tee-Object -FilePath $masterLog -Append
py -3.11 -m gridfm_datakit.cli generate user_config_case2000_s20000_droop_strong_p10.yaml 2>&1 | Tee-Object -FilePath $log
'Finished user_config_case2000_s20000_droop_strong_p10.yaml at ' + (Get-Date) | Tee-Object -FilePath $masterLog -Append
$log = Join-Path $logDir 'run_p20_$ts.log'
'Starting user_config_case2000_s20000_droop_strong_p20.yaml at ' + (Get-Date) | Tee-Object -FilePath $masterLog -Append
py -3.11 -m gridfm_datakit.cli generate user_config_case2000_s20000_droop_strong_p20.yaml 2>&1 | Tee-Object -FilePath $log
'Finished user_config_case2000_s20000_droop_strong_p20.yaml at ' + (Get-Date) | Tee-Object -FilePath $masterLog -Append
$log = Join-Path $logDir 'run_p40_$ts.log'
'Starting user_config_case2000_s20000_droop_strong_p40.yaml at ' + (Get-Date) | Tee-Object -FilePath $masterLog -Append
py -3.11 -m gridfm_datakit.cli generate user_config_case2000_s20000_droop_strong_p40.yaml 2>&1 | Tee-Object -FilePath $log
'Finished user_config_case2000_s20000_droop_strong_p40.yaml at ' + (Get-Date) | Tee-Object -FilePath $masterLog -Append
'Batch end: ' + (Get-Date) | Tee-Object -FilePath $masterLog -Append

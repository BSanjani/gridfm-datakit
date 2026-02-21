$ErrorActionPreference = 'Stop'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDir = 'logs_case2000_s20000_lowmem'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$masterLog = Join-Path $logDir ("master_lowmem_$ts.log")
"Batch start: $(Get-Date)" | Tee-Object -FilePath $masterLog

$configs = @(
  'user_config_case2000_s20000_droop_strong_p05.yaml',
  'user_config_case2000_s20000_droop_strong_p10.yaml',
  'user_config_case2000_s20000_droop_strong_p20.yaml',
  'user_config_case2000_s20000_droop_strong_p40.yaml'
)

foreach ($cfg in $configs) {
  $tag = ($cfg -replace '^.*_p(\d\d)\.yaml$','$1')
  $runLog = Join-Path $logDir ("run_p${tag}_$ts.log")
  "Starting $cfg at $(Get-Date)" | Tee-Object -FilePath $masterLog -Append
  py -3.11 -m gridfm_datakit.cli generate $cfg 2>&1 | Tee-Object -FilePath $runLog
  "Finished $cfg at $(Get-Date)" | Tee-Object -FilePath $masterLog -Append
}
"Batch end: $(Get-Date)" | Tee-Object -FilePath $masterLog -Append

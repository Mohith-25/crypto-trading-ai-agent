$env:BINANCE_API_KEY="your_binance_api_key_here"
$env:BINANCE_API_SECRET="your_binance_api_secret_here"
$env:BINANCE_TESTNET="true"
$env:TRADING_MODE="SPOT"
$env:HF_TOKEN="your_hf_token_here"

Write-Host "Starting main.py (FastAPI OpenEnv Server) in the background..."
$serverProcess = Start-Process -FilePath "python" -ArgumentList "main.py" -PassThru -NoNewWindow
Start-Sleep -Seconds 5

Write-Host "Starting inference.py (Agent)..."
python inference.py

Write-Host "Finished inference. Shutting down the background server..."
Stop-Process -Id $serverProcess.Id

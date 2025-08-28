 
function Get-FreePort {
    $ports = 8888..8899  
    foreach ($port in $ports) {
        $test = Test-NetConnection -ComputerName "localhost" -Port $port
        if ($test.TcpTestSucceeded -eq $false) {
            Write-Host "Jupyter Got port : $port"
            return $port
        }
    }
    return $null
}
 
$port = Get-FreePort
if ($port -eq $null) {
      exit
}
 
 
docker run -dit --user root -p 8888:8888 -v C:/tim:/home/jovyan/work jupyter/base-notebook start-notebook.sh --NotebookApp.token='' --NotebookApp.allow_root=True
docker run -dit --user root -p 8888:8888 -v C:/tim:/home/jovyan/work jupyter/base-notebook start-notebook.sh --NotebookApp.token='' --NotebookApp.allow_root=True --NotebookApp.ip='0.0.0.0'
Write-Host "Jupyter Notebook http://localhost:$port"
docker run -dit --user root --network host -v C:/tim:/home/jovyan/work jupyter/base-notebook start-notebook.sh --NotebookApp.token='' --NotebookApp.allow_root=True --NotebookApp.ip='0.0.0.0'
docker exec -it --user root 5e35f88 /bin/bash
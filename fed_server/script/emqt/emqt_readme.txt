step  1
cd C:\tim\aicam\main\fed_server
.\activate
step 2
(.grpc_env) PS C:\tim\aicam\main\fed_server> python .\emqx_manager.py

netstat -ano | findstr :18083
step 3 check the serer activate or not
username : admin
password : tim0726@

for conda
step 1
conda env list
conda activate my_env
(base) PS C:\Users\Administrator> conda activate my_env
step 2
cd C:\tim\aicam\main\fed_server\
(my_env) PS C:\tim\aicam\main\fed_server> python .\emqx_manager.py

Notice:
1)It must be showing on screen
"Starting EMQX server..."
2) The emqt actually run on
c:\emqt\emqx-5.0.26-windows-amd64\bin\emqx start -c c:\emqt\emqx-5.0.26-windows-amd64\etc\emqx.conf
EMQX started. Dashboard: http://localhost:18083 (admin/public)
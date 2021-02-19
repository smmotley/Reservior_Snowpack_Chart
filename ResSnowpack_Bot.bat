set err.log=U:\Documents\Programming\PycharmProjects\Prod\SnowLevel_Bot\err.log
Rem set python=C:\Users\smotley\AppData\Local\Continuum\anaconda3\envs\Data_Grabber\python.exe
@echo off
For /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
For /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
@echo Current Date And Time Is: %mydate%_%mytime% >> U:\Documents\Programming\PycharmProjects\Prod\Reservior_Snowpack_Chart\err.log 2>&1
@echo Starting...>> U:\Documents\Programming\PycharmProjects\Prod\Reservior_Snowpack_Chart\err.log 2>&1
call activate Data_Grabber
python U:\Documents\Programming\PycharmProjects\Prod\Reservior_Snowpack_Chart\Reservoir_Status.py >> U:\Documents\Programming\PycharmProjects\Prod\Reservior_Snowpack_Chart\err.log 2>&1
@echo FINISHED
@echo *****************************************************************
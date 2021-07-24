# SignLanguageConverter
Sign Language to Audio Converter

# Dependencies
1. Python 3.8 or above
2. Python dependencies (numpy, mediapipe, cv2)
3. Sql Server
4. UIPath Apps

# Assets to be created in Orchestrator
1. CSVFilePath (Path of the CSV file containing dataset of recorded symbols) - C:\SignLanguageInterpreter\gesture_train.csv (You can put your own path)
2. Frames (No of continuous frames required to capture a sign)- 25 (You can change no as per your requirement)
3. FunctionName -detect (You can change no as per your requirement)
4. IdleFrames (No of idle frames required between 2 words) - 6
5. PythonPath (Path of Python Installation) - C:\Users\XXXX\AppData\Local\Continuum\miniconda3\envs\TF (You can put your own path)
6. ScriptPath (Path of Python script file) - C:\SignLanguageInterpreter\Script\singleHand.py (You can put your own path)
7. SqlServerName ( Name of the SQL Server) - ITEM-S56784\SQLEXPRESS (You can put your own ServerName)


# Python-Deep-Learning-Library
* Make sure of PATH variables
* Commands
* Logic

### Make sure of PATH variables
About -> 
Advanced system settings -> 
Environment variables ->
Edit PATH variable ->
Add three paths if not present:
* C:\Users\user_name\AppData\Local\Programs\Python\Python39 (python)
* C:\Users\Mikolaj Semeniuk\AppData\Local\Programs\Python\Python39\Scripts (pip)
* C:\Users\Mikolaj Semeniuk\AppData\Local\Programs\Python\Python39 (name you will get after running `pip install --user pipenv`)
### Commands
```sh
pip list

pip install --user pipenv
pip install --user --upgrade pipenv

pip install mypy

cd ai
pipenv shell
pipenv install numpy

mypy library/ --ignore-missing-imports

pipenv run python .\library\xor.py
```

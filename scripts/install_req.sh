source venv/Scripts/activate

pip install git+https://github.com/Deathdadev/Orpheus-Speech-PyPi --extra-index-url https://download.pytorch.org/whl/cu121
pip install accelerate

pip install -r requirements.txt

pip install --upgrade numpy==1.26.4
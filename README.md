# segment-images
Conda enviroment:
`conda create -n segment-image-env`

E depois ative o ambiente criado:
`conda activate segment-image-env`


Para instalar as bibliotecas necessárias antes de rodar o código:
`pip3 install -r requirements.txt`

É necessário fazer o download do modelo do SAM (prieira versão) e adicioná-lo na pasta './checkpoints' !!!

Para rodar o código, digite no terminal:
`python3 src/main.py`

Nos próximos releases será utilizado o modelo SAM 2.

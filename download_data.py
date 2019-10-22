import os

languages = ['python', 'java', 'go', 'php', 'ruby', 'javascript']

os.system('mkdir data')

for language in languages:
    os.system(f'wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip')
    os.system(f'unzip {language} -d data')
    os.system(f'rm {language}.zip')
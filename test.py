#%%
from bs4 import BeautifulSoup
from requests import get

# %%
url = 'https://www.defenseindustrydaily.com/f18-hornet-fleets-keeping-em-flying-02816/#2014–2020'
response = get(url)
if response.status_code==200:
    soup = BeautifulSoup(response.text, 'lxml')
    results = soup.find_all(text=True)
    text = ''
    for result in results:
     text = text + ' ' + result
    print(text[:1000])
# %%

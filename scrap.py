import requests
from bs4 import BeautifulSoup


def scrap(url):
    print('starting to scrap.........')
    webpage=requests.get(url)
    soup=BeautifulSoup(webpage.content,'lxml')
    soup.prettify()
    story_section=soup.find_all('section',class_='story-section')

    for content in story_section:
        paragraphs=content.find_all('p')
    
    main_content=[]
    for paragraph in paragraphs:
        main_content.append(paragraph.text)

    cleaned_content="\n".join(main_content)
    print('content scrapped')
    return cleaned_content


    
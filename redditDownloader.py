import os
import re
import requests
import praw
import concurrent.futures
import getpass


class redditImageScraper:
    def __init__(self):
        self.client_id = str(getpass.getpass(
            "For Client ID and Client secret go to\n https://www.reddit.com/prefs/apps \n and create your token\nEnter Your Client ID:\n"))

        self.client_secret = str(getpass.getpass("\nEnter your client Secret:\n"))
        # print(client_id,client_secret)
        self.reddit = praw.Reddit(client_id=self.client_id,
                                  client_secret=self.client_secret,
                                  user_agent='my user agent')

    def download(self, image):
        r = requests.get(image['url'])
        with open(image['fname'], 'wb') as f:
            f.write(r.content)
        print("File Downloaded: {} File Path: {} ".format(image['url'], image['fname']))
        self.images['filepath'].append(image['fname'])

    def start(self, path, page):
        self.subreddit = self.reddit.subreddit(page)
        if page == 'Dark_memes':
            self.subreddit.quaran.opt_in()
        self.imageGen = self.subreddit.top(limit=2000)
        self.images = {'filepath': []}
        image = []
        print('Started')
        try:

            for submission in self.imageGen:

                if submission.url.endswith(('jpg', 'jpeg', 'png')):

                    fname = path + re.search('(?s:.*)\w/(.*)', submission.url).group(1)
                    if not os.path.isfile(fname):
                        image.append({'url': submission.url, 'fname': fname})

            if len(image):
                if not os.path.exists(path):
                    os.makedirs(path)
                with concurrent.futures.ThreadPoolExecutor() as ptolemy:
                    ptolemy.map(self.download, image)
        except Exception as e:
            if e == 'received 403 HTTP response':
                print("Page is quarantined or you have'nt joined it yet")


downloader = redditImageScraper()

downloader.start(path='./dataset/negative/',page  = 'Dark_memes')
downloader.start(path='./dataset/positive/', page= 'wholesomememes')
downloader.start(path='./dataset/neutral/', page = 'antimeme')
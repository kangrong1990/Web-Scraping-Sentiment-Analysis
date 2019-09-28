from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import os
import pathlib
import glob
import time
import pickle
import numpy as np


class hotel_scraper(object):

    def __init__(self, homepage, url, page_no):
        self.homepage = homepage
        self.url = url
        self.page_no = page_no
        self.data_dir = self.get_dir('data/')
        self.model_dir = self.get_dir('models/')

    def get_dir(self, filename=''):
        cwd = os.getcwd()
        return cwd + '/' + filename

    def gen_urllist(self):
        url_list = []
        for p in range(60, 30 * (self.page_no + 2), 30):
            url = 'https://www.tripadvisor.ca/Hotels-g155019-oa' + repr(p) + '-Toronto_Ontario-Hotels.html'
            url_list.append(url)
        return url_list

    def gen_soup(self, url):
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        return soup

    def get_titles_links(self):
        title_list = []
        link_list = []
        url_list = self.gen_urllist()
        for url in url_list:
            soup = self.gen_soup(url)
            titles = soup.find_all('div', class_='listing_title')
            for t in titles:
                title_list.append(t.get_text())
                link = homepage + t.a.get('href')
                link_list.append(link)
        return title_list, link_list

    def get_prices(self):
        price_list = []
        url_list = self.gen_urllist()
        for url in url_list:
            soup = self.gen_soup(url)
            prices = soup.find_all('div', class_="price-wrap")
            for p in prices:
                price = p.find('div', class_='price autoResize').get_text()
                price_list.append(price)
        return price_list

    def get_hotel_info(self):
        address_list = []
        phone_list = []
        _, link_list = self.get_titles_links()
        for link in link_list:
            soup = self.gen_soup(link)
            address = soup.find('span', class_='public-business-listing-ContactInfo__ui_link--1_7Zp public-business-listing-ContactInfo__level_4--3JgmI')
            phone = soup.find('span', class_='public-business-listing-ContactInfo__nonWebLinkText--nGymU public-business-listing-ContactInfo__ui_link--1_7Zp public-business-listing-ContactInfo__level_4--3JgmI')
            add = address
            if address:
                add = address.get_text()
            ph = phone
            if phone:
                ph = phone.get_text()
            address_list.append(add)
            phone_list.append(ph)
        return address_list, phone_list

    def hotel_data(self):
        if os.path.isfile(self.data_dir + 'hotel.csv'):
            print('Hotel Data Exists!')
        else:
            cols = ['Name', 'Link', 'Address', 'Phone', 'Price']
            title_list, link_list = self.get_titles_links()
            price_list = self.get_prices()
            address_list, phone_list = self.get_hotel_info()
            df = pd.DataFrame(columns=cols)
            df['Name'] = pd.Series(title_list)
            df['Link'] = pd.Series(link_list)
            df['Address'] = pd.Series(address_list)
            df['Phone'] = pd.Series(phone_list)
            df['Price'] = pd.Series(price_list)
            df.to_csv(self.data_dir + 'hotel.csv', index=False)

    def load_csv(self, file):
        if os.path.isfile(file):
            df = pd.read_csv(file)
            return df
        else:
            print('File Does Not Exist!')

    def get_reviews(self):
        review_dir = self.data_dir + 'reviews/'
        df = self.load_csv(self.data_dir + 'hotel.csv')
        link_list = df['Link']
        hotel_list = df['Name']
        for hotel, link in zip(hotel_list, link_list):
            start = time.time()
            hotel_reviews = []
            hotel_ratings = []
            all_reviews_ratings = {}
            review_urls = [link] + [('or' + repr(i)).join(link.split('Reviews')) for i in range(5, 5 * 1500, 5)]
            for url in review_urls:
                soup = self.gen_soup(url)
                if soup.find('div', class_='ui_alert warning hotels-review-list-parts-ReviewListView__no_results_alert--3kljL'):
                    break
                else:
                    reviews, ratings = self.hotel_reviews(url)
                    hotel_ratings.append(ratings)
                    hotel_reviews.append(reviews)
            hotel_reviews = [r for l in hotel_reviews for r in l]
            hotel_ratings = [r for l in hotel_ratings for r in l]
            all_reviews_ratings['Reviews'] = hotel_reviews
            all_reviews_ratings['Ratings'] = hotel_ratings
            end = time.time()
            print('{0} Reviews Found for {1} ({2:.3f}s)'.format(len(hotel_reviews), hotel, end - start))
            with open(review_dir + hotel, 'wb') as file:
                pickle.dump(all_reviews_ratings, file)

    def hotel_reviews(self, url):
        soup = self.gen_soup(url)
        ratings = soup.find_all('div', class_='hotels-review-list-parts-RatingLine__bubbles--1oCI4')
        reviews = soup.find_all('q', class_='hotels-review-list-parts-ExpandableReview__reviewText--3oMkH')
        ratings_per_hotel = []
        reviews_per_hotel = []
        for r in ratings:
            rating = float(r.span.get('class')[-1].split('_')[-1]) / 10
            ratings_per_hotel.append(rating)
        for r in reviews:
            reviews_per_hotel.append(r.get_text())
        return reviews_per_hotel, ratings_per_hotel

    def load_reviews(self):
        review_dir = self.data_dir + 'reviews/'
        review_list = glob.glob(review_dir + '*')
        review_dict = {}
        for review in review_list:
            with open(review, 'rb') as file:
                name = review.split('/')[-1]
                review_dict[name] = pickle.load(file)
        return review_dict

    def gen_review_data(self):
        if os.path.isfile(self.data_dir + 'review_data.csv'):
            print('Review Data Exists!')
        else:
            file = self.data_dir + 'hotel.csv'
            df = self.load_csv(file)
            review_dict = self.load_reviews()
            hotel_list = list(review_dict.keys())
            cols = ['Name', 'Address', 'Phone', 'Price', 'Reviews', 'Ratings']
            finalDF = pd.DataFrame(columns=cols)
            for name in hotel_list:
                row = df[df['Name'] == name]
                reviews = review_dict[name]
                reviewDF = pd.DataFrame(reviews)
                tempDF = pd.DataFrame({c: np.repeat(row[c].values, len(reviewDF)) for c in df.columns})
                reviewDF = pd.concat([reviewDF, tempDF], axis=1)
                reviewDF = reviewDF[cols]
                finalDF = finalDF.append(reviewDF)
            finalDF = finalDF.sample(frac=1).reset_index(drop=True)
            finalDF.to_csv(self.data_dir + 'review_data.csv', index=False)


if __name__ == '__main__':
    homepage = 'https://www.tripadvisor.ca'
    url = 'https://www.tripadvisor.ca/Hotels-g155019-Toronto_Ontario-Hotels.html'
    page_no = 2
    scraper = hotel_scraper(homepage, url, page_no)
    scraper.hotel_data()
    scraper.gen_review_data()

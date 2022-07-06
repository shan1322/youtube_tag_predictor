import os

import requests
import pandas as pd
from data_load_save_pandas import save_csv, load_csv
import time
import numpy as np


class DataApi:
    def __init__(self, base_uri, api_key, directory):
        self.base_uri = base_uri
        self.api_key = api_key
        self.directory = directory

    def get_video_ids(self, search_keyword, directory, data_per_search=150):
        search_uri = self.base_uri + "search"
        next_page = None
        payload = {"part": "snippet",  # default value snippet
                   "type": "video",
                   "key": self.api_key,
                   "q": search_keyword,  # q is query
                   "maxResults": 50,
                   "pageToken": next_page

                   }
        base_response = requests.get(search_uri, params=payload)
        if base_response.status_code == 200:
            base_response = base_response.json()
            next_page = base_response["nextPageToken"]
            base_data = [[videos["id"]["videoId"], videos["snippet"]['title'], videos["snippet"]["description"]] for
                         videos in base_response["items"]]
            data_size = 0
            while data_size <= data_per_search:
                payload['pageToken'] = next_page
                response = requests.get(search_uri, params=payload)
                if response.status_code == 200:
                    response = response.json()
                    next_page = response["nextPageToken"]
                    data = [[videos["id"]["videoId"], videos["snippet"]['title'], videos["snippet"]['description']]
                            for
                            videos in response["items"]]
                    base_data.extend(data)
                    data_size += 50
                else:
                    print(response.json())
            df = pd.DataFrame(base_data, columns=["id", "title", "description"])
            df["search_word"] = search_keyword
            file_name = search_keyword.replace(" ", "_") + ".csv"
            save_csv(directory, file_name, df)



        else:
            print(base_response.json())

    def combine_data(self, directory):
        files = os.listdir(directory)
        combine = []

        for file in files:
            df = load_csv(path=directory, file_name=file)
            combine.append(df)
        combine_df = pd.concat(combine)
        print(combine_df.shape)
        save_csv(path=directory, file_name="combine_data.csv", df=combine_df)

    def get_stats_tags_50(self, df):
        ids = list(df.id)
        id_list = ids.copy()
        ids = ",".join(ids)
        # https://www.googleapis.com/youtube/v3/videos?channelTitle&id=sTPtBvcYkO8,HuK2WPmcbjA&key=AIzaSyB6vUPEGtpfaYxCOV16wrBpJalXFsfbhp4&part=snippet,statistics&fields=items(snippet(tags)),items(statistics(viewCount,likeCount))
        payload = {"part": "snippet,statistics",
                   "key": self.api_key,
                   "id": ids,
                   # "fields": "items(snippet(tags)),items(statistics(viewCount,likeCount,dislikeCount,favoriteCount,commentCount))"
                   }
        response = requests.get(self.base_uri + "videos", params=payload)
        if response.status_code == 200:
            response = response.json()
            items = response["items"]
            for index in range(len(items)):
                item = items[index]
                id = id_list[index]
                tags = ""
                if "tags" in item["snippet"].keys():
                    tags = item["snippet"]["tags"]
                tags = ",".join(tags)
                view_count, comment_count, favorite_count, like_count = 0, 0, 0, 0
                if "statistics" in item.keys():
                    if "viewCount" in item["statistics"].keys():
                        view_count = item["statistics"]["viewCount"]
                    if "likeCount" in item["statistics"].keys():
                        like_count = item["statistics"]["likeCount"]
                    if "favoriteCount" in item["statistics"].keys():
                        favorite_count = item["statistics"]["favoriteCount"]
                    if "commentCount" in item["statistics"].keys():
                        comment_count = item["statistics"]["commentCount"]
                    df.loc[df["id"] == id, ["viewCount", "likeCount", "favoriteCount",
                                            "commentCount",
                                            "tags"]] = view_count, like_count, favorite_count, comment_count, tags
            del df['Unnamed: 0']
            del df['Unnamed: 0.1']
            del df['Unnamed: 0.2']
            return df
        else:
            print(response.text)
            return pd.DataFrame()

    def get_stats_tag(self, file_name):
        dfs = []
        df = load_csv(file_name="combine_data.csv", path=self.directory)
        columns = ["tags", "viewCount", "likeCount", "favoriteCount", "commentCount"]
        for column in columns:
            df[column] = ""
        ###rint(self.get_stats_tags_50(df=df.loc[51:100]))
        for i in range(0, len(df), 45):
            df_stage = df.loc[i:i + 45]
            df_out = self.get_stats_tags_50(df=df_stage)
            dfs.append(df_out)
        df = pd.concat(dfs)
        save_csv(path=self.directory, file_name=file_name, df=df)

    def split(self, file_name):
        df = load_csv(path=self.directory, file_name=file_name)
        df = df[["title", "tags"]]
        train_df = df[df["tags"].notnull()]
        predict_df = df[df["tags"].isna()]
        save_csv(path=self.directory, file_name="train_tag.csv", df=train_df)
        save_csv(path=self.directory, file_name="predict_tag.csv", df=predict_df)




"""dataapi.get_video_ids(search_keyword="wwe", directory="data")
dataapi.get_video_ids(search_keyword="indian history", directory="data")
dataapi.get_video_ids(search_keyword="ashoka", directory="data")
dataapi.get_video_ids(search_keyword="rss", directory="data")
dataapi.get_video_ids(search_keyword="movie ", directory="data")
dataapi.get_video_ids(search_keyword="3 idiots", directory="data")
"""
# dataapi.combine_data(directory="data")
# dataapi.get_stats_tag(file_name="stat_tag_new.csv")
if __name__ == "__main__":
    dataapi = DataApi(base_uri="https://youtube.googleapis.com/youtube/v3/", api_key=os.environ['api_key'],
                      directory="data")
    dataapi.split(file_name="stat_tag_new.csv")

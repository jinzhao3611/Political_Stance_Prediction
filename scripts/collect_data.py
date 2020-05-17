# this is used to collect data from YouTube API
import json
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os


class YouTubeComments(object):
    def __init__(self, service_name, service_version, api_key):
        """

        :param service_name: google api service name, in this case "youtube"
        :param service_version: api version, in this case "v3"
        :param api_key: api key
        """
        self.service_name = service_name
        self.service_version = service_version
        self.service = build(
            self.service_name, self.service_version, developerKey=api_key
        )
        self.video_info = None

    def keyword_search(
        self, keyword, out_folder, policy, surname, max_items=5, stats=False, to_csv=False
    ):
        """

        :param keyword: query string in youtube search box
        :param max_items: returned item (video, channel, youtuber) number
        :param stats: Apart from v_id and v_title, also including
                common statistics (viewCount,likeCount,dislikeCount,favoriteCount,commentCount)
        :param to_csv: output a video_stats_{keyword}.amt_input_csv file
        :return:
        """
        raw_results = (
            self.service.search()
            .list(q=keyword, type="video", part="id,snippet", maxResults=max_items)
            .execute()
        )
        basic_v_info = {}
        for result in raw_results.get("items", []):
            if result["id"]["kind"] == "youtube#video":
                basic_v_info[result["id"]["videoId"]] = result["snippet"]["title"]
        if stats:
            self.video_info = self._get_stats(basic_v_info)
        else:
            self.video_info = [dict(v_id=k, v_title=v) for k, v in basic_v_info]
        if to_csv:
            self.info2csv(
                out_file=f"{out_folder}/video_stats_{policy}_{surname}.amt_input_csv"
            )

    def _get_stats(self, basic_v_info):
        """
        helper function to get the video statistics
        :param basic_v_info:
        :return:
        """
        v_ids = ",".join(basic_v_info.keys())
        videos_stats = (
            self.service.videos().list(id=v_ids, part="id,statistics,snippet").execute()
        )
        video_info = []
        for i in videos_stats["items"]:
            temp = dict(v_id=i["id"], v_title=basic_v_info[i["id"]])
            temp.update(i["statistics"])
            temp.update(i["snippet"])
            video_info.append(temp)
        return video_info

    def info2csv(self, out_file):
        """
        output video info into amt_input_csv
        :param out_file:
        :return:
        """
        df = pd.DataFrame.from_dict(self.video_info)
        df.to_csv(out_file, index=False)
        print(f"Save {out_file} successfully!")

    def get_video_comments(
        self, v_id, out_folder, max_c=20, verbose=False,
    ):
        """
        given a video id, return a list of comments of the video
        :param v_id: unique youtube video id
        :param max_c: returned number of the comments
        :param verbose: output the raw comments information into json file
        :return: a list of comments of a video
        """
        comments = []
        try:
            results = (
                self.service.commentThreads()
                .list(
                    part="snippet",
                    videoId=v_id,
                    textFormat="plainText",
                    order="time",
                    maxResults=max_c,
                )
                .execute()
            )
            if verbose:
                with open(
                    f"{out_folder}/{v_id}_raw_comments.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(results, f, indent=4)
            for item in results["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                print("Comment: " + comment)
        except HttpError:
            print("Comment disabled by http error!")
        except UnicodeEncodeError:
            print("Encoding error!")
        return comments

    def get_all_comments(self, out_folder, max_per_v=100):
        """
        output the comments and basic videos info into json
        :param max_per_v: returned comment number of each video
        :param out_file:
        :return:
        """
        v_info = [(d["v_id"], d["v_title"]) for d in self.video_info]
        v_ids = [i for i, _ in v_info]
        v_titles = [i for _, i in v_info]
        out = {}
        for i, v_id in enumerate(v_ids):
            comments = self.get_video_comments(
                v_id, out_folder, max_c=max_per_v, verbose=True
            )
            out[str(i)] = {"id": v_id, "title": v_titles[i], "comments": comments}
        with open("{}/all_comments.json".format(out_folder), "w") as f:
            json.dump(out, f, indent=4)
            print(f"Finish dumping {out_folder}/all_comments.json!")


def create_folder(policy, candidates):
    path1 = policy
    os.mkdir(path1)
    for candidate in candidates:
        path2 = "{}/{}_{}".format(policy, policy, candidate)
        try:
            os.mkdir(path2)
        except OSError:
            print("Creation of the directory %s failed" % path2)
        else:
            print("Successfully created the directory %s " % path2)


def main(policy):
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    DEVELOPER_KEY = "AIzaSyAsx2fWYRy2l1Qu7X7WUMfCT3G8vM0Nlhs"
    cjy = YouTubeComments(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, DEVELOPER_KEY)

    candidates = {
        "biden": "Joe Biden",
        "klobuchar": "Amy Klobuchar",
        "buttigieg": "Pete Buttigieg",
        "sanders": "Bernie Sanders",
        "warren": "Elizabeth Warren",
        "bloomberg": "Michael Bloomberg",
        "Gabbard": "Tulsi Gabbard"

    }

    create_folder(policy, list(candidates.keys()))

    for surname, fullname in candidates.items():
        out_folder = "{}/{}_{}".format(policy, policy, surname)
        cjy.keyword_search(
            "{} policy {}".format(policy, fullname),
            out_folder,
            policy,
            surname,
            max_items=10,
            stats=True,
            to_csv=True,
        )
        cjy.get_all_comments(out_folder)


if __name__ == "__main__":
    policies = [
        "abortion",
        "economic",
        "education",
        "environment",
        "gun",
        "healthcare",
        "immigration",
        "LGBTQ",
    ]

    for policy in policies:
        main(policy)

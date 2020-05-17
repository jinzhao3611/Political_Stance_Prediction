import pandas as pd
import json
import re
import demoji
import csv
import emoji as emj

demoji.download_codes()


def contain_link(sent: str) -> bool:
    url = re.findall(
        "http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]| [! * \(\),] | (?: %[0-9a-fA-F][0-9a-fA-F]))+",
        sent,
    )
    return True if url else False


# def contain_csource_emoji(sent:str) -> bool:
#     pattern = '\\u.{4}\\u.{4}'
#     csource_emoji = re.findall(pattern, sent)
#     return True if csource_emoji else False
# return re.sub(pattern, '', sent) # this returns re.error: incomplete escape \u at position 0, it's an issue with python 3.7


def get_media_list(policy, candidate):
    media_list = list()
    with open(
        "{}/{}_{}/video_stats_{}_{}.amt_input_csv".format(
            policy, policy, candidate, policy, candidate
        ),
        "r",
    ) as csvfile:
        data = csv.reader(csvfile)
        media_index = ""
        for i, row in enumerate(data):
            if i == 0:
                media_index = row.index("channelTitle")
            else:
                media_list.append(row[media_index])
    return media_list


def json2csv_all(policy):
    candidates = {
        "biden": "Joe Biden",
        "klobuchar": "Amy Klobuchar",
        "buttigieg": "Pete Buttigieg",
        "sanders": "Bernie Sanders",
        "warren": "Elizabeth Warren",
        "bloomberg": "Michael Bloomberg",
        "Gabbard": "Tulsi Gabbard"
    }
    dfs = list()
    for candidate in candidates:
        dfs.append(
            json2csv(get_media_list(policy, candidate), policy, candidate=candidate)
        )

    # shuffle and get 400 data from each candidate
    frames = [df.sample(frac=1) for df in dfs]
    # frames = [sanders_df.sample(frac=1), biden_df.sample(frac=1), buttigieg_df.sample(frac=1), klobuchar_df.sample(frac=1), warren_df.sample(frac=1)]
    all_healthcare_df = pd.concat(frames).sample(frac=1).iloc[:1500]
    print(all_healthcare_df.iloc)

    output_file = "amt_input_csv/{}_{}.amt_input_csv".format(policy, "all")
    all_healthcare_df.to_csv(output_file, index=False)


def json2csv(media, policy, candidate):
    input_file = "{}/{}_{}/all_comments.json".format(policy, policy, candidate)
    with open(input_file, "r") as f:
        data = json.load(f)

    all_comments = list()
    video_titles = list()
    media_list = list()

    for i, vid in enumerate(data.items()):
        comments = vid[1]["comments"]
        url_free_comments = list()
        for comment in comments:
            if not contain_link(comment):
                url_free_comments.append(comment)

        all_comments.extend(url_free_comments)
        video_titles.extend([vid[1]["title"]] * len(url_free_comments))
        media_list.extend([media[i]] * len(url_free_comments))

    data = {
        "video_title": video_titles,
        "policy": [policy] * len(all_comments),
        "media": media_list,
        "text": all_comments,
    }

    df = pd.DataFrame(data, columns=["video_title", "policy", "media", "text"])
    # output_file = 'amt_input_csv/{}_{}.amt_input_csv'.format(policy, candidate)
    # df.to_csv(output_file, index=False)
    return df


def parse_emoji(sentence: str, sub: bool):
    emojis = demoji.findall(sentence)
    for emoji in emojis:
        if sub:
            desc = emj.demojize(emoji)
            desc = re.sub(r"[:_]", " ", desc).strip()
            sentence = sentence.replace(emoji, f" EMOJI:[{desc}] ")
        else:
            sentence = sentence.replace(emoji, f" {emoji} ")
    sentence = re.sub(r"\s{2,}", " ", sentence)

    return sentence


if __name__ == "__main__":
    policy = "abortion"

    json2csv_all(policy)
    with open("amt_input_csv/{}_all.amt_input_csv".format(policy), "r") as csvfile:
        data = csv.reader(csvfile)
        output = list()
        for row in data:
            flag = False
            # for ele in row:
            #     emojis = demoji.findall(ele)
            #     if emojis:
            #         flag = True
            if not flag:
                output.append(row)

    with open(
        "amt_input_csv/{}_all_emoji_processed.amt_input_csv".format(policy), "w"
    ) as f:
        writer = csv.writer(f)
        for i, row in enumerate(output):
            if i == 0:
                writer.writerow(row)
            else:
                new_row = row[:3]
                text = parse_emoji(row[3], True)
                new_row.append(text)
                writer.writerow(new_row)

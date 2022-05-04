import botometer
import pandas
import pandas as pd
import matplotlib.pyplot as plt

rapidapi_key = "01ab1a9140msh0f5cb516ded2eb8p1a45c8jsna6d145388643"

twitter_app_auth = {
    'consumer_key': 'N1bGI6OvvxdMG490oSb0fhDlk',
    'consumer_secret': '8HpTasH6V5Jm5NW5iy9t1x6fZ8WKUsMlAhCHF93UuOIiKvwVey'
   # 'access_token': '942830173854011392-T72PnBswgFdxZLvk8zR7xbk0EmsTl9X',
    #'access_token_secret': 'XITuOFdRlSaUNgo4JWsH7sXyUqpnF2cmsqatSb9xFwUh9',
}
bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)
training_set = pd.read_csv(r'C:\Users\Esau\PycharmProjects\PersonalProject\split_training_set (version 2).csv')
ids = training_set['id'].tolist()

for element in ids:
    if type(element) == 'float':
        element = str(int(float(element)))
    elif type(element) == 'string':
        element == str(element)

types = training_set['type'].tolist()
human_scores = []
bot_scores = []
organization_scores = []

for id, result in bom.check_accounts_in(ids):
    try:
        if result["user"]["majority_lang"] == "en":
            if (training_set.iat[ids.index(id),1]).lower() == "human":
                human_scores.append(result['display_scores']['english']['overall'])
            elif (training_set.iat[ids.index(id),1]).lower() == "bot":
                bot_scores.append(result['display_scores']['english']['overall'])
            elif (training_set.iat[ids.index(id), 1]).lower() == "organization":
                organization_scores.append(result['display_scores']['english']['overall'])
        else:
            if (training_set.iat[ids.index(id), 1]).lower() == "human":
                human_scores.append(result['display_scores']['universal']['overall'])
            elif (training_set.iat[ids.index(id), 1]).lower() == "bot":
                bot_scores.append(result['display_scores']['universal']['overall'])
            elif (training_set.iat[ids.index(id), 1]).lower() == "organization":
                organization_scores.append(result['display_scores']['universal']['overall'])

        print(f'{id} has been processed')
    except Exception as e:
        print("{} Could not be found {}".format(id, e))


human_scores_df = pandas.DataFrame({'human_scores': human_scores})
human_scores_df.to_csv('human scores')

bot_scores_df = pandas.DataFrame({'bot_scores': bot_scores})
bot_scores_df.to_csv('bot scores')

organization_scores_df = pandas.DataFrame({'organization_scores': organization_scores})
organization_scores_df.to_csv('organization scores')

human_hist = human_scores_df.hist()
plt.savefig('human_hist.png')

bot_hist = bot_scores_df.hist()
plt.savefig('bot_hist.png')

organization_hist = organization_scores_df.hist()
plt.savefig('organization_hist.png')

plt.show()
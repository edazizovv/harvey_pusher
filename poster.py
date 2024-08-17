#
import vk_api
import telebot


#
def joly(strey):
    res = ''
    for j in range(len(strey)):
        res = strey[-j - 1] + res
        if (j + 1) % 3 == 0:
            res = ' ' + res
    return res


def forma(bases, predictions, from_date, to_date):
    return '#DailyForecast\n' \
           'Now ({0}) -> Forecast ({1})\n\n'.format(from_date, to_date) + \
           'Close Prices\n' + '\n' \
        .join(['{0}:\t{1} -> {2}'.format('BTC/USD', joly('{0:.0f}'.format(bases['BTC-USD'])), joly('{0:.0f}'.format(predictions['BTC-USD']))),
               '{0}:\t{1} -> {2}'.format('ETH/USD', joly('{0:.0f}'.format(bases['ETH-USD'])), joly('{0:.0f}'.format(predictions['ETH-USD']))),
               '{0}:\t{1:.4f} -> {2:.4f}'.format('XRP/USD', bases['XRP-USD'], predictions['XRP-USD']),
               '{0}:\t{1:.2f} -> {2:.2f}'.format('LTC/USD', bases['LTC-USD'], predictions['LTC-USD']),
               '{0}:\t{1:.4f} -> {2:.4f}'.format('DOGE/USD', bases['DOGE-USD'], predictions['DOGE-USD'])]) + '\n' \
            '📈 Dashboard'


def _forma(statuses, predoooo, eldy):
    return '#DailyForecast\n' \
           'Now -> Forecast (24 hour)\n\n' \
           'Close Prices\n' + '\n' \
        .join(['{0}:\t{1} -> {2}'.format('BTC/USD', joly('{0:.0f}'.format(eldy['BTC-USD'])), joly('{0:.0f}'.format(predoooo['BTC-USD']))),
               '{0}:\t{1} -> {2}'.format('ETH/USD', joly('{0:.0f}'.format(eldy['ETH-USD'])), joly('{0:.0f}'.format(predoooo['ETH-USD']))),
               '{0}:\t{1:.4f} -> {2:.4f}'.format('XRP/USD', eldy['XRP-USD'], predoooo['XRP-USD']),
               '{0}:\t{1:.2f} -> {2:.2f}'.format('LTC/USD', eldy['LTC-USD'], predoooo['LTC-USD']),
               '{0}:\t{1:.4f} -> {2:.4f}'.format('DOGE/USD', eldy['DOGE-USD'], predoooo['DOGE-USD'])]) + '\n' \
            '📈 Dashboard'


def vk_post(statuses, predoooo, eldy):
    owner_id = ''
    from_group = 1
    message = forma(statuses, predoooo, eldy)
    # publish_date =
    guid = 'haha'

    client_id_source = 'C:/Users/Edward/Desktop/vk_auth.txt'
    crs = open(client_id_source, "r")
    for columns in (raw.strip().split() for raw in crs):
        login = columns[0]
        pw = columns[1]

    vk_session = vk_api.VkApi(login, pw)
    vk_session.auth()

    vk = vk_session.get_api()

    vk.wall.post(owner_id=owner_id, from_group=from_group, message=message,  # publish_date=publish_date,
                 guid=guid)


def _tg_post(statuses, predoooo, eldy):
    client_id_source = 'C:/Users/Edward/Desktop/tg_token.txt'
    crs = open(client_id_source, "r")
    for columns in (raw.strip().split() for raw in crs):
        tg_token = columns[0]

    bot = telebot.TeleBot(tg_token, parse_mode=None)

    chat_id = '@thethcry'
    message = message = forma(statuses, predoooo, eldy)
    bot.send_message(chat_id=chat_id, text=message)


def tg_post(bases, predictions, from_date, to_date):
    client_id_source = 'C:/Users/Edward/Desktop/tg_token.txt'
    crs = open(client_id_source, "r")
    for columns in (raw.strip().split() for raw in crs):
        tg_token = columns[0]

    bot = telebot.TeleBot(tg_token, parse_mode=None)

    chat_id = '@thethcry'
    message = message = forma(bases, predictions, from_date, to_date)
    bot.send_message(chat_id=chat_id, text=message)

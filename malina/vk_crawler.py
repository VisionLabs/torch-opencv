import os
import re

import requests
import json

import time
import urllib

vk_group = "scandinaviaclub"

# 1000 is max
def full_name(user_object):
    return str(user_object["uid"]) + "_" + user_object["first_name"] + "_" + user_object["last_name"]


def get_user_and_store_picture(user):
    time.sleep(request_timeout)
    json_user = json.loads(requests.get(get_user_by_id_url + str(user))._content)
    user_object = json_user["response"][0]

    pattern_bdate = re.compile("\d{1,2}\.\d{1,2}\.\d\d\d\d")

    # if user_object["deactivated"] != "deleted" or user_object["deactivated"] != "banned" or user_object["sex"] == 1:

    if "bdate" in user_object and \
                pattern_bdate.match(user_object["bdate"]) and \
                (2015 - int(user_object["bdate"].split(".")[2])) in xrange(18, 36) and \
                user_object["sex"] == 1 and \
                user_object["photo_max_orig"] != "http://vk.com/images/camera_400.png": # no photo
        urllib.urlretrieve(user_object["photo_max_orig"], photo_folder + "/" + full_name(user_object) + ".jpg")
        time.sleep(request_timeout)

        print "Parsed person:"
        print user_object
        print
    else:
        print "Person is rejected:"
        print user_object
        print

users_count = "1000"
initial_offset = 12000

offset = initial_offset
request_timeout = 0.0

for i in range(1, 1000):

    get_users_in_group_url = "https://api.vk.com/method/groups.getMembers?" \
                             "group_id=" + vk_group + \
                             "&count=" + users_count + \
                             "&offset=" + str(offset)
    get_user_by_id_url = "https://api.vk.com/method/users.get?fields=bdate,sex,photo_max_orig&user_ids="

    photo_folder = "result/" + vk_group
    if not os.path.isdir(photo_folder):
        os.makedirs(photo_folder)

    response = requests.get(get_users_in_group_url)
    json_response = json.loads(response._content)

    users = json_response["response"]["users"]

    for user in users:
        for i in xrange(1, 5):
            try:
                get_user_and_store_picture(user)
                break
            except Exception as e:
                print "Exception occurred:"
                print e
                print "Waiting for 5 sec"
                if i == 4:
                    print "WARNING: we reject to analyze the profile because 5 attempts were made, the user is below"
                    print user
                time.sleep(5)


    offset += 1000

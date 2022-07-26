import csv
import shelve
import hashlib
import ipaddress
import json
import re
import socket
import sys

import tldextract


def policheck_is_first_party(package_name, dest_domain, privacy_policy):
    # Get start of packagename com.benandow.policylint --> com.benandow
    splitPackageName = package_name.split(u'.')
    rPackageName = u'{}.{}'.format(splitPackageName[0], splitPackageName[1])

    # Get root destination domain (reversed) (e.g., policylint.benandow.com --> com.benandow)
    splitDestDom = dest_domain.split(u'.')
    if len(splitDestDom) < 2:
        return False
    rDestDomRev = u'{}.{}'.format(splitDestDom[-1], splitDestDom[-2])

    # Check if root dest_domain (reversed) matches start of package_name
    if rPackageName == rDestDomRev:
        return True

    # Check if root privacy_policy url (reversed) matches start of package name
    if privacy_policy != u'NULL' and len(privacy_policy) > 0:
        #Reverse root policy URL: https://www.benandow.com/privacy --> com.benandow
        splitDom = re.sub(r'/.*$', '', re.sub(r'http(s)?://', u'', privacy_policy, flags=re.I)).split(u'.')
        rPolUrlRev = u'{}.{}'.format(splitDom[-1], splitDom[-2])
        # Check if the root privacy policy url matches the destination domain..
        if rPolUrlRev == rDestDomRev:
            return True
    return False


def main():
    input_csv, dns_cache, output_json = sys.argv[1:]

    FLOW_CSV_COLUMNS = ['package_name', 'app_name', 'version_name', 'version_code',
                        'data_type', 'dest_domain', 'dest_ip', 'arb_number', 'privacy_policy']
    DATA_MAP = {
        'aaid': 'advertising id',
        'fingerprint': None,
        'androidid': 'android id',
        'geolatlon': 'precise geolocation',
        'hwid': 'serial number',
        'routerssid': 'router ssid',
        'routermac': 'mac address',
        'imei': 'imei',
        'phone': 'phone number',
        'email': 'email address',
        'wifimac': 'mac address',
        'invasive': None,
        'package_dump': 'package dump',
        'simid': 'sim serial number',
        'real_name': 'person name',
        'gsfid': 'gsf id'
    }

    data = {}

    with open(input_csv, encoding="utf-8") as fin:
        reader = csv.DictReader(fin, fieldnames=FLOW_CSV_COLUMNS)
        ip_to_domain = dict()

        for row in reader:
            if (dtype := DATA_MAP[row["data_type"]]) is None:
                continue

            package_name = row["package_name"]
            dest_domain = row["dest_domain"]
            dest_ip = row["dest_ip"]
            privacy_policy_url = row["privacy_policy"]
            privacy_policy_id = hashlib.blake2s(privacy_policy_url.encode()).hexdigest()

            if dest_domain != "":
                ip_to_domain[dest_ip] = dest_domain

            if package_name not in data:
                data[package_name] = {
                    "app_name": row["app_name"],
                    "privacy_policy_url": privacy_policy_url,
                    "privacy_policy_id": privacy_policy_id,
                    "flows": []
                }

            data[package_name]["flows"].append({
                "data_type": dtype,
                "dest_domain": dest_domain,
                "dest_ip": dest_ip,
            })

    # Fill in missing domains via reverse lookups
    # Use shelve as cache because reverse lookup can be slow
    with shelve.open(dns_cache) as cache:
        error_ips = set()

        for package_name, info in data.items():
            flow_list = info["flows"]

            for flow in flow_list:
                dest_ip = flow["dest_ip"]

                if flow["dest_domain"] != "" or dest_ip in error_ips:
                    continue

                if dest_ip in ip_to_domain:
                    flow["dest_domain"] = ip_to_domain[dest_ip]
                else:
                    try:
                        if dest_ip in cache:
                            dest_domain = cache[dest_ip]
                        else:
                            dest_domain, _ = socket.getnameinfo((dest_ip, 0), 0)
                    except socket.gaierror:
                        dest_domain = dest_ip
                        print("ERROR:", dest_ip)

                    cache[dest_ip] = dest_domain

                    try:
                        ipaddress.ip_address(dest_domain)
                    except ValueError:
                        flow["dest_domain"] = ip_to_domain[dest_ip] = dest_domain
                        print(dest_ip, "->", dest_domain)
                    else:
                        print("UNKNOWN IP:", dest_ip)

    # Fill in party label after we get all domains
    for package_name, info in data.items():
        privacy_policy_url = info["privacy_policy_url"]

        for flow in info["flows"]:
            if policheck_is_first_party(package_name, flow["dest_domain"], privacy_policy_url):
                flow["party"] = "first party"
            else:
                flow["party"] = "third party"

    with open(output_json, "w", encoding="utf-8") as fout:
        json.dump(data, fout)


if __name__ == "__main__":
    main()

import csv
import functools
import hashlib
import importlib.resources as pkg_resources
import ipaddress
import json
import re
import shelve
import socket
import sys

import tldextract
import poligrapher

CLOUD_PROVIDERS = {
    "amazonaws.com", "amazonaws.com.cn", "cloudfront.net", "akamaitechnologies.com", "linode.com",
    "herokuapp.com", "googleusercontent.com", "fastly.net", "cloudflare.com"
}


class DomainMapper:
    def __init__(self, entity_info_json):
        self.domain_map = {}

        with open(entity_info_json, encoding="utf-8") as fin:
            for entity, entity_info in json.load(fin).items():
                self.domain_map.update({dom: entity for dom in entity_info["domains"]})

    @functools.lru_cache
    def is_first_party(self, package_name, dest_domain, privacy_policy_url):
        dest_tld = tldextract.extract(dest_domain)

        if dest_tld.suffix != "":
            # Major cloud service: Assume the developer controls the data
            if dest_tld.registered_domain in CLOUD_PROVIDERS:
                return True

            # Main domain name exists in the package name
            # eg: com.promobitech.mobilock.pro -> mobilock.in
            if dest_tld.domain in package_name:
                return True

            dest_domain_owner = self.domain_map.get(dest_tld.registered_domain)
        else:
            return False

        rev_package_name = ".".join(reversed(package_name.split(".")))
        first_party_keywords = set()

        package_name_tld = tldextract.extract(rev_package_name)
        package_name_owner = self.domain_map.get(package_name_tld.registered_domain)

        # Package name and dest domain have the same owner entity
        # eg: com.twitter.android -> twimg.com
        if package_name_owner and package_name_owner == dest_domain_owner:
            return True

        # Reversed package_name is a domain. Add the domain part to keyword list
        if package_name_tld.suffix != "":
            first_party_keywords.add(package_name_tld.domain.lower())

        policy_url_tld = tldextract.extract(privacy_policy_url)

        # Add captalized word in the package name to keyword list
        if m := re.search(r"\w+\.([A-Z]\w+)", package_name):
            first_party_keywords.add(m[1].lower())


        # Add the domain part of policy url to keyword list
        if policy_url_tld.suffix != "":
            # There are many policies host on third-party domain
            # I don't think this is reliable but just to align with PoliCheck
            first_party_keywords.add(policy_url_tld.domain.lower())

        for keyword in first_party_keywords:
            if re.search(rf"\b{keyword}", dest_domain, re.I):
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
        'geolatlon': 'geolocation',
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

    with pkg_resources.path(poligrapher, "extra-data") as extra_data:
        entity_info = extra_data / "entity_info.json"

    domain_mapper = DomainMapper(entity_info)
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
            if domain_mapper.is_first_party(package_name, flow["dest_domain"], privacy_policy_url):
                flow["party"] = "first party"
            else:
                flow["party"] = "third party"

    with open(output_json, "w", encoding="utf-8") as fout:
        json.dump(data, fout)


if __name__ == "__main__":
    main()

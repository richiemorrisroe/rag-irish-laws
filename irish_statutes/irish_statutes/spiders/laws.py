import datetime as dt
import json
from pathlib import Path


import scrapy


class LawsSpider(scrapy.Spider):
    name = "laws"
    allowed_domains = ["www.irishstatutebook.ie"]
    start_urls = ["https://www.irishstatutebook.ie/"]

    def start_requests(self):
        MIN_YEAR = 2000
        MAX_YEAR = dt.datetime.now().year
        BASE_ACT_URL = "https://www.irishstatutebook.ie/eli/{year}/act/"
        urls = [BASE_ACT_URL.format(year=year) for year in range(MIN_YEAR, MAX_YEAR)]
        self.log(f"{urls=}")
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        self.log(f"{response=}")
        year = response.url.split("/")[-3]
        tbls = response.css('div.acts-datatables')
        public_acts = response.css('#public-acts')
        links, names = public_acts.css('a::attr(href)').getall(), public_acts.css('a::text').getall()
        links_html = [x for x in links if x.endswith('.html')]
        name_link_dict = {name: link for name, link in zip(names, links_html)}
        name_link_dict['year'] = year
        yield name_link_dict

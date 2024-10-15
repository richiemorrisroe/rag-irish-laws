import datetime as dt
import re
from urllib.parse import urljoin

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
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        year = response.url.split("/")[-3]
        public_acts = response.css('#public-acts')
        links, names = public_acts.css('a::attr(href)').getall(), public_acts.css('a::text').getall()
        links_html = [x for x in links if x.endswith('.html')]
        
        for name, link in zip(names, links_html):
            transformed_url = self.transform_url(link, year)
            if transformed_url:
                yield scrapy.Request(transformed_url, callback=self.parse_act, meta={'name': name, 'year': year})

    def transform_url(self, original_url, year):
        # Extract the act number from the original URL
        match = re.search(r'/(\d+)/index\.html$', original_url)
        if match:
            act_number = match.group(1)
            # Construct the new URL
            new_url = f"/eli/{year}/act/{act_number}/enacted/en/print.html"
            return urljoin(self.start_urls[0], new_url)
        return None

    def parse_act(self, response):
        name = response.meta['name']
        year = response.meta['year']
        
        # Extract the full text of the act
        full_text = response.css('body').extract_first()
        
        yield {
            'name': name,
            'year': year,
            'url': response.url,
            'full_text': full_text
        }

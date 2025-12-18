import requests
import json
import os
import re
import numpy as np

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_API_URL = os.getenv("SERPER_API_URL", "https://google.serper.dev/search")


class SearchSerper:
    def __init__(self, max_results=3) -> None:
        self._max_results = max_results

    def search(self, query):
        payload = json.dumps({"q": query, "gl": "us"})
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        response = requests.request(
            "POST", SERPER_API_URL, headers=headers, data=payload
        )
        results = response.json()
        try:
            results = results["organic"]
        except KeyError:
            results = []
        res = [
            (
                result["snippet"] if "snippet" in result else "",
                result["link"] if "link" in result else "",
            )
            for result in results
            if "snippet" in result
        ]
        result = {x[1]: x[0] for x in res}
        return result

    async def search_queries(self, description, country_of_origin_name, output_currency, exchange_rate_usd, uom, is_fallback: bool = False):
        # Get search results
        search_queries = extract_search_queries(description, country_of_origin_name)
        google_results, prices_from_search = {}, []
        for query in search_queries:
            google_results.update(self.search(query))

        if google_results:
            google_results = [{"text": context, "index": index, "url": url} 
                             for index, (url, context) in enumerate(google_results.items())]
            price_res = await extract_prices_google_ai(output_currency, exchange_rate_usd, uom, description, google_results)

            for res in price_res:
                try:
                    price = res["price"]
                    price = float(re.findall(r'[\d\-\.]+', str(price))[0])
                    prices_from_search.append({
                        "price": price,
                        "url": google_results[res["index"]]['url'],
                    })
                except Exception:
                    continue
        if not prices_from_search:
            logger.info("No search results from google")
            raise ValueError("No search results from google")

        prices = [x['price'] if '-' not in x else np.mean([float(y) for y in x.split('-')]) for x in prices_from_search]
        Q1, Q3 = np.percentile(prices, 25), np.percentile(prices, 75)
        IQR = Q3 - Q1
        filtered = [x for x in prices_from_search if Q1 - 1.5 * IQR <= x['price'] <= Q3 + 1.5 * IQR]

        return filtered, [x['price'] for x in filtered], {"message": "[Search Google Fallback] Successfully calculated the customs value"} if is_fallback else {"message": "[Search Google Prefer] Successfully calculated the customs value"}

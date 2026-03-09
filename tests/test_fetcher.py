import unittest

from arxiv_llm_watch.fetcher import ArxivFetcher


class FetcherTests(unittest.TestCase):
    def test_build_query_uses_categories_only_when_keywords_missing(self) -> None:
        query = ArxivFetcher.build_query(["cs.CL", "cs.AI"])
        self.assertEqual(query, "cat:cs.CL OR cat:cs.AI")

    def test_build_query_combines_categories_with_keywords(self) -> None:
        query = ArxivFetcher.build_query(["cs.CL"], keywords=["reasoning", "alignment"])
        self.assertEqual(query, '(cat:cs.CL) AND (all:"reasoning" OR all:"alignment")')


if __name__ == "__main__":
    unittest.main()

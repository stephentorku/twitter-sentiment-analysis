import unittest
import requests


class MyTestCase(unittest.TestCase):
    endpoint_url = "http://127.0.0.1:5000/get_data"
    web_app_url = "http://127.0.0.1:5000/"

    def test_get_endpoint(self):
        g_test = requests.get(MyTestCase.endpoint_url)
        self.assertEqual(len(g_test.json()), 1)


    def test_urls(self):
        g_test = requests.get(MyTestCase.endpoint_url)
        i_test = requests.get(MyTestCase.web_app_url)
        self.assertEqual(g_test.status_code, 200)
        self.assertEqual(i_test.status_code, 200)


if __name__ == '__main__':
    unittest.main()

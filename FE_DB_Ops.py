import sqlite3
import pandas as pd

from Hyper_Setup import log_file

import logging
logging = logging.getLogger("main")

class DB_Ops:
    def __init__(self, company_list_file = "companylist.csv", black_list_file = 'blacklist.csv', db_folder = 'Databases/'):

        logging.info("Loading %s %s", company_list_file, black_list_file)
        comp_list = list(pd.read_csv(company_list_file).Symbol)
        black_list = list(pd.read_csv(black_list_file).Symbol)

        i = 0
        for k in comp_list:
            if k in black_list:
                logging.info("%s blacklisted, Popping", k)
                comp_list.pop(i)
            i += 1

        self.company_list = comp_list
        self.db_folder = db_folder

        max_value = 0
        for k in comp_list:
            values_list = self.get_values_company(k)
            if max_value < len(values_list):
                max_value = len(values_list)

        logging.info("%s total rows", max_value)


        self.max_rows = max_value


    def get_list_companies(self):
        return self.company_list


    def get_values_company(self, company_sym):
        list_stocks = sqlite3.connect(self.db_folder + company_sym + ".db")
        cursor = list_stocks.cursor()
        table_name = "data"

        cursor.execute("SELECT * FROM %s" % (table_name))

        return [float(k[1]) for k in cursor.fetchall()]


    def get_max_rows(self):
        return self.max_rows


    def get_companies_count(self):
        return len(self.company_list)
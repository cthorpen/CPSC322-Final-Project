from mysklearn import myutils
import copy
import csv
from tabulate import tabulate

# from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return [len(self.data), len(self.column_names)]

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = []
        try:
            col_idx = self.column_names.index(col_identifier)
            for row in self.data:
                value = row[col_idx]
                # col.append(value)
                if include_missing_values == True:
                    col.append(value)
                else:
                    if value != "NA":
                        col.append(value)
        except ValueError:
            print(col_identifier, "is not a valid column identifier")
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.column_names)):
                try:
                    num_val = float(self.data[i][j])
                    self.data[i][j] = num_val
                except ValueError:
                    # print(self.data[i][j], "couldn't be converted to a float")
                    x = 1
        pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        for i in range(len(self.data), -1, -1):
            if i in row_indexes_to_drop:
                if i < len(self.data):  # might be unneccessary, bc loop is going down
                    self.data.pop(i)
                    i -= 1
        pass

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            self.column_names = next(csvreader)
            for row in csvreader:
                self.data.append(row)
        self.convert_to_numeric()
        csvfile.close()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        for i in range(len(self.column_names) - 1):
            outfile.write(str(self.column_names[i]) + ",")
        outfile.write(str(self.column_names[-1]) + "\n")
        for row in self.data:
            for j in range(len(row) - 1):
                outfile.write(str(row[j]) + ",")
            outfile.write(str(row[-1]) + "\n")
        outfile.close()
        pass

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        duplicates = []
        # get indexes of key_column_names
        key_idx = self.get_key_column_name_indices(key_column_names)
        # create table of only the values (columns) being checked
        val_table = self.get_values_to_check(key_idx)
        # check for dups here
        for row in val_table:
            for i in range(val_table.index(row) + 1, len(val_table)):
                if val_table[i] == row:
                    if i not in duplicates:
                        duplicates.append(i)
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_data = []
        for row in self.data:
            if '' or ' ' not in row:
                new_data.append(row)
        self.data = new_data
        pass

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        column = self.get_column(col_name)
        for row in self.data:
            for i in range(len(row)):
                if row[i] == "NA" or row[i] == "":
                    avg = myutils.get_average(column)
                    row[i] = round(avg, 1)
        pass

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        stats_column_names = ["attribute",
                              "min", "max", "mid", "avg", "median"]
        stats_data = []
        self.convert_to_numeric()
        # check empty
        if len(self.data) > 0:
            for i in range(len(col_names)):
                # compute stats for each column provided
                row = []
                column = self.get_column(col_names[i])
                # attribute
                row.append(col_names[i])
                # min
                row.append(min(column))
                # max
                row.append(max(column))
                # mid
                row.append((row[1] + row[2]) / 2)
                # average
                row.append(myutils.get_average(column))
                # median
                row.append(myutils.get_median(column))
                stats_data.append(row)
        return MyPyTable(stats_column_names, stats_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table = []
        joined_headers = self.join_headers(
            self.column_names, other_table.column_names)
        # find table indices of keys in each table
        t1_header_idx = self.find_header_index(
            self.column_names, key_column_names)
        t2_header_idx = self.find_header_index(
            other_table.column_names, key_column_names)
        for row1 in self.data:
            for row2 in other_table.data:
                # check keys in each row match
                if self.check_row_match(row1, row2, t1_header_idx, t2_header_idx):
                    # join rows
                    new_row = self.join_rows(
                        row1, row2, self.column_names, other_table.column_names)
                    joined_table.append(new_row)
        return MyPyTable(joined_headers, joined_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        joined_table = []
        joined_header = self.join_headers(
            self.column_names, other_table.column_names)
        # find table indices of keys in each table
        t1_header_idx = self.find_header_index(
            self.column_names, key_column_names)
        t2_header_idx = self.find_header_index(
            other_table.column_names, key_column_names)
        joined_header_index = self.find_header_index(
            joined_header, key_column_names)
        for row1 in self.data:
            is_match = False
            for row2 in other_table.data:
                if self.check_row_match(row1, row2, t1_header_idx, t2_header_idx):
                    is_match = True
                    # join rows
                    new_row = self.join_rows(
                        row1, row2, self.column_names, other_table.column_names)
                    joined_table.append(new_row)
            if is_match == False:
                # an item in self.data is not in othertable.data
                # fill missing values with "NA" on the row and add it
                new_row = self.fill_missing_values(
                    row1, self.column_names, joined_header)
                joined_table.append(new_row)

        for row in other_table.data:
            if not self.check_table_match(row, joined_table, t2_header_idx, joined_header_index):
                new_row = self.fill_missing_values(
                    row, other_table.column_names, joined_header)
                joined_table.append(new_row)

        return MyPyTable(joined_header, joined_table)  # TODO: fix this

    # UTILITY FUNCTIONS TO HELP CLEAN UP

    def fill_missing_values(self, row, old_header, joined_header):
        '''Utility function
        '''
        new_row = []
        for head in joined_header:
            if head in old_header:
                new_row.append(row[old_header.index(head)])
            else:
                new_row.append("NA")
        return new_row

    def check_table_match(self, row_to_check, joined_table, index, joined_index):
        '''Utility function
        '''
        for joined_rows in joined_table:
            if self.check_row_match(row_to_check, joined_rows, index, joined_index):
                return True
        return False

    def check_row_match(self, row1, row2, index1, index2):
        """Utility function
        """
        row1_key = []
        for idx in index1:
            row1_key.append(row1[idx])
        row2_key = []
        for idx in index2:
            row2_key.append(row2[idx])
        if row1_key == row2_key:
            return True
        else:
            return False

    def find_header_index(self, header, comp_key):
        """Utility function
        """
        table_header_index = []
        for key in comp_key:
            if key in header:
                table_header_index.append(header.index(key))
        return table_header_index

    def get_composite_key(self, header1, header2):
        """Utility function
        """
        keys = []
        for col1 in header1:
            for col2 in header2:
                if col1 == col2:
                    keys.append(col1)
        return keys

    def join_headers(self, header1, header2):
        """Utility function
        """
        new_header = copy.copy(header1)
        for head in header2:
            if head not in header1:
                idx = header2.index(head)
                new_header.append(header2[idx])
        return new_header

    def join_rows(self, row1, row2, header1, header2):
        """Utility function
        """
        new_row = copy.copy(row1)
        for col in header2:
            if col not in header1:
                idx = header2.index(col)
                new_row.append(row2[idx])
        return new_row

    def get_key_column_name_indices(self, key_column_names):
        """Utility function
        """
        key_idx = []
        for key_name in key_column_names:
            key_idx.append(self.column_names.index(key_name))
        return key_idx

    def get_values_to_check(self, key_idx):
        """Utility function
        """
        val_table = []
        for row in self.data:
            row_val = []
            for item in row:
                if row.index(item) in key_idx:
                    row_val.append(row[row.index(item)])
            val_table.append(row_val)
        return val_table

    def get_frequencies(self, col_name):
            col = MyPyTable.get_column(self, col_name)
            values = []
            counts = []
            try:
                col.sort()
                for value in col:
                    if value in values:
                        counts[-1] += 1 # okay because col is sorted
                    else: # haven't seen this value before
                        values.append(value)
                        counts.append(1)
            except:
                for value in col:
                    if value in values:
                        counts[values.index(value)] += 1 # okay because col is sorted
                    else: # haven't seen this value before
                        values.append(value)
                        counts.append(1)
            
            return values, counts # we can return multiple items
            # packaged into a tuple
            
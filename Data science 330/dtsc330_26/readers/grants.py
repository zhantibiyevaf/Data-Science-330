#grants
import pandas as pd


class Grants():
    def __init__(self, path: str):
        """Create and parse a Grants file

        Args:
            path (str): the location of the file on the disk
        """
        self.path = path
        self.df = self._parse(path)

    def _parse(self, path: str):
        """Parse a grants file"""
        df = pd.read_csv(path)
        
        mapper = {
            'APPLICATION_ID': 'application_id', 
            'BUDGET_START': 'start_at',
            'PROJECT_START': 'project_start',
            'ACTIVITY': 'grant_type',
            'TOTAL_COST': 'total_cost',
            'PI_NAMEs': 'pi_names', 
            'ORG_NAME': 'organization',
            'ORG_CITY': 'city',
            'ORG_STATE': 'state',
            'ORG_COUNTRY': 'country', 
        }
        # make column names lowercase
        # maybe combine for budget duration?
        df = df.rename(columns=mapper)[mapper.values()]
        df["start_at"] = df["start_at"].fillna(df["project_start"])
        return df
    
    def get(self):
        """Get parsed grants"""
        return self.df        


if __name__ == '__main__':
    # This is for debugging
    grants = Grants('Data science 330/data/RePORTER_PRJ_C_FY2025.csv')

    df1 = grants.get()
    print(df1)
    print(df1['start_at'].isna().sum())
    print(df1[df1["start_at"].isna()])
    print(df1['application_id'].nunique())
    print(df1['project_start'].isna().sum())

    # show rows where project_start is NA
    print(df1[df1['project_start'].isna()]) # hm so these don't event have countries? i think i can drop them
    df1 = df1.drop(df1[df1['project_start'].isna()].index)
    print(df1['project_start'].isna().sum()) # now the dates are all done

    # starting question 2 here
    df2 = df1.copy()
    # need to remove (contact) from the names that have it
    df2['pi_names'] = df2['pi_names'].str.replace('(contact)', '')
    df2['pi_names'] = df2['pi_names'].str.split(';')
    #df2['pi_names'] = df2['pi_names'].str.replace('(contact)', '')
    print(df2['pi_names']) 
    # ^ this now is the list of names, that i should separate and turn into many rows with single values
    # i will use .explode() for this
    print(df2[['application_id', 'pi_names']].head(10))
    df2 = df2.explode('pi_names')
    print(df2[['application_id', 'pi_names']])
    #checking the specific id number that i knew had multiple names
    print(df2[df2['application_id'] == 10308732][['application_id', 'pi_names']])

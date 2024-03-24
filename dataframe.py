import pandas as pd

from sklearn.model_selection import train_test_split

class Init_dataframe:
    def __init__(self, csv_path) -> None:
        df_original = self.read_csv_and_encode_label(csv_path)
        df_original, df_undup = self.get_undup_df(df_original)
        df_val = self.get_val_df(df_undup)
        df_train = self.get_train_df(df_original, df_val)

        self.df_train = df_train.reset_index()
        self.df_val = df_val.reset_index()

    def read_csv_and_encode_label(self, csv_path):
        df_original = pd.read_csv(csv_path)
        df_original["encoded_dx"] = pd.Categorical(df_original["dx"]).codes

        return df_original

    def get_undup_df(self, df: pd.DataFrame):
        df_original = df.copy()

        df_undup = df_original.groupby("lesion_id").count()
        df_undup = df_undup[df_undup["image_id"] == 1]
        df_undup.reset_index(inplace=True)

        unique_list = list(df_undup["lesion_id"])

        df_original["duplicates"] = df_original["lesion_id"]
        df_original["duplicates"] = df_original["duplicates"].apply(
            lambda x: "unduplicated" if x in unique_list else "duplicates"
        )

        return df_original, df_undup
    
    def get_val_df(self, df: pd.DataFrame) -> pd.DataFrame:
        y = df['encoded_dx']
        _, df_val = train_test_split(df, test_size=0.2, random_state=101, stratify=y)
        
        return df_val
    
    def get_train_df(self, df: pd.DataFrame, df_val: pd.DataFrame):
        df_original = df.copy()

        val_list = list(df_val['image_id'])

        df_original['train_or_val'] = df_original['image_id']
        df_original['train_or_val'] = df_original['train_or_val'].apply(lambda x: 'val' if str(x) in val_list else 'train')

        df_train = df_original[df_original['train_or_val'] == 'train']

        data_aug_rate = [15,10,5,50,0,40,5]
        for i in range(7):
            if data_aug_rate[i]:
                df_train=df_train.append([df_train.loc[df_train['encoded_dx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
        
        return df_train
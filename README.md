## Code

- data_preprocess.py: preprocess data for Dependent-T5.
- parameters.py: define parameters of Dependent-T5.
- train_tools.py: traning and evaluation code of Dependent-T5.
- nli.py: run it to train and test mT5-Text Matching for LJP.
- main.py: run it to train and test Dependent T5 for LJP. 

## Data

- article_content_dict.pkl: the dict of law article contents and the key is law article item.

- data_{train, valid, test}.json: 

  ```json
  {
      "fact": "fact description.",
      "interpretation": "court view",
      "meta":{
          "relevant_articles": [the list of violated law articles items],
          "accusation": [the list of charges],
          "term_of_imprisonment": {
              "death_penalty": True or False,
              "life_imprisonment": True or False,
              "imprisonment": months of imprisoment,
          }
      }
      
  }
  ```

  


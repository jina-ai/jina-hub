# SpellChecker

This crafter provides spelling correction capabilities by wrapping a model based on `PyngramSpell` which is a model based on bigrams.
The crafter is implemented with the capability to use a `BKTree` structure to have an efficient search of candidates.

The crafter requires a model to be fit in order to be loaded by the `SpellChecker` to perform query correction.

## Usage

Users can use Pod images in several ways:

1. Run with Flow API
   ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my_crafter', uses='docker://jinahub/pod.crafter.spellchecker:0.0.1-1.1.10', port_in=55555, port_out=55556))
    ```

2. Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: spellchecker
      uses: crafters/nlp/SpellChecker/config.yml
  ```
  
  and then in `spellchecker.yml`:
  ```yaml
  !SpellChecker
  with:
    model_path: 'model.pickle'
  ```


3. Run with Jina CLI
   ```bash
    jina pod --uses docker://jinahub/pod.crafter.spellchecker:0.0.1-1.1.10 --port-in=55555 --port-out 55556
    ```

   Conventional local usage with `uses` argument
    ```bash
    jina pod --uses crafters/nlp/SpellChecker/config.yml --port-in 55555 --port-out 55556
    ```
    
4. Run with Docker (`docker run`)
   ```bash
    docker run --rm -p 55555:55555 -p 55556:55556 jinahub/pod.crafter.spellchecker:0.0.1-1.1.10 --port-in 55555 --port-out 55556
    ```

## Simple Usage

   ```python
    """ train model """
    from pyngramspell import PyNgramSpell
    train_data = ['expected correct sentences from a corpus', 'more expected words']
    speller = PyNgramSpell(min_freq=0)
    speller.fit(train_data)
    speller.save('model.pickle')
    
    """ instantiate crafter """
    from jina.hub.crafters.nlp.SpellChecker import SpellChecker
    crafter = SpellChecker(model_path='model.pickle')
    assert crafter.craft('xpected correct setences')['text'] == 'expected correct sentences'
    ```

# PySpellChecker

This crafter provides spelling correction capabilities by wrapping [pyspellchecker](https://github.com/barrust/pyspellchecker).

It is based on a pure python implementation based on Peter Norvig's [blog](https://norvig.com/spell-correct.html) post on setting up a simple spell checking algorithm.
## Usage

Users can use Pod images in several ways:

1. Run with Flow API
   ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my_crafter', uses='docker://jinahub/pod.crafter.pyspellchecker:0.0.1-1.1.7', port_in=55555, port_out=55556))
    ```

2. Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: pyspellchecker
      uses: crafters/nlp/PySellChecker/config.yml
  ```
  
  and then in `pyspellchecker.yml`:
  ```yaml
  !PySellChecker
  with:
    language: en
    case_sensitive: False
  ```


3. Run with Jina CLI
   ```bash
    jina pod --uses docker://jinahub/pod.crafter.pyspellchecker:0.0.1-1.1.7 --port-in=55555 --port-out 55556
    ```

   Conventional local usage with `uses` argument
    ```bash
    jina pod --uses crafters/nlp/PySpellChecker/config.yml --port-in 55555 --port-out 55556
    ```
    
 4. Run with Docker (`docker run`)
   ```bash
    docker run --rm -p 55555:55555 -p 55556:55556 jinahub/pod.crafter.pyspellchecker:0.0.1-1.1.7 --port-in 55555 --port-out 55556
    ```


## Simple example

``` python
crafter = PySpellChecker(langauge='en')
input = np.array(['wrng sentenca']
crafter_output = crafter.craft(input)[0]
assert crafter_output['text'] == 'wrong sentence'
```

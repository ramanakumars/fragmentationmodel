[<img src="https://app.readthedocs.org/projects/fragmentationmodel/badge/?style=flat-square">](https://fragmentationmodel.readthedocs.io/en/latest/index.html)

# Fragmentation Model
Parameterized meteor fragmentation model written in Python based on 
[Avramenko et al. 2014](https://doi.org/10.1002/2013JD021028) and 
[Wheeler et al. 2019](https://doi.org/10.1016/j.icarus.2017.02.011)


### Test cases
Validation cases are given in the `tests/` directory. Run the original Chelyabinsk test case in
Avramenko et al. 2014 using:

```
	python3 -m tests.run_test_av2014
```

in the main repo directory, 
which plots the energy deposition profile and ratio of ablated mass (Fig 4 and 7 in the paper)


The Sankar et al. 2020 jovian impact case can be run using 

```
	python3 -m tests.run_all_sankar2020
```

which plots the energy released for all test cases (Fig 7 in the paper). 

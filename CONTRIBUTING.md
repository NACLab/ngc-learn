How to Contribute
=================

We are very happy to see you reading this page!

Our team wholeheartedly and excitedly welcomes the community to contribute to ngc-learn. Contributions from members of the community will help ensure the long-term success of this project and achieve its goal to create useful tools for others to
conduct research in neurobiologically-motivated neural systems and algorithms. However, before you plan to make contributions, here are important resources to get started with:

- Read the ngc-learn [documentation](https://ngc-learn.readthedocs.io/en/latest/#) and its [source paper](https://www.nature.com/articles/s41467-022-29632-7#:~:text=Neural%20generative%20models%20can%20be,predictive%20processing%20in%20the%20brain.)
- Check our latest status from existing [issues](https://github.com/ago109/ngc-learn/issues), [pull requests](https://github.com/ago109/ngc-learn/pulls), and [branches](https://github.com/ago109/ngc-learn/branches) and avoid duplicate efforts
- <!--Join our [NGC-LEARN Slack](https://ngc-learn.slack.com) workspace for technical discussions.--> Please [email us](mailto:ago@cs.rit.edu) to be added to our Slack workspace for technical discussions.

We encourage the community to make major types of contributions including:

- **Bug fixes**: Address open issues and fix bugs presented in the `master` branch (or find and fix currently unknown bugs).
- **Documentation fixes**: Improving the documentation is just as important as improving the library itself. If you find a typo in the documentation, or have made improvements, do not hesitate to send an [email](mailto:ago@cs.rit.edu) or preferably submit a GitHub pull request. Documentation can be found under the [docs](https://github.com/ago109/ngc-learn/tree/master/docs) directory (**note** that you will need to have installed all of the items in the specific `requirements.txt` file in `/docs/` folder in order to build the Read the Docs documentation files).
- **Model museum designs:** Implement new/classical predictive processing models (that appear in both modern/older publications or have been used in useful/interesting applications or analyses) using ngc-learn's nodes and cables and integrate them into the Model Museum, i.e., [model museum](https://github.com/ago109/ngc-learn/tree/master/ngclearn/museum). Also [notify us](mailto:ago@cs.rit.edu) if you would like the NAC Laboratory to internally schedule integrating and including your published/used predictive processing model or neurobiological system in the [Model Museum](https://ngc-learn.readthedocs.io/en/latest/museum/model_museum.html).
- **Extensions to the core engine:** Incorporate new [nodes](https://github.com/ago109/ngc-learn/tree/master/ngclearn/engine/nodes), including new/missing functionalities/aspects of neural dynamics, error neuron simulations, dendritic tree calculations, and optimizations, or [cables](https://github.com/ago109/ngc-learn/tree/master/ngclearn/engine/cables), including new/missing synaptic transformations (such as complex or more neurobiologically-faithful transmission of vector signals) or synaptic update rules (such a multi-factor Hebbian rules, neuromodulation, or spike-time dependent plasticity).
- **Additional assets:** Incorporate new [models](https://github.com/ago109/ngc-learn/tree/master/ngclearn/museum) (which would also go into the [Model Museum](https://ngc-learn.readthedocs.io/en/latest/museum/model_museum.html)) that are not based in predictive processing, for example, models/systems based on contrastive Hebbian learning or equilibrium propagation, contrastive divergence, or competitive learning. This can include any missing functionality in the neural dynamics simulation that would support other aspects of neurobiological processing.
- **New functionalities:** Implement new features including, but not limited to, rendering/visualization tools, [data generating processes](https://github.com/ago109/ngc-learn/tree/master/ngclearn/generator) to test/evaluate neural systems on, mathematical functions/distribution utilities helpful for analyzing or facilitating the calculations underlying simulated neural dynamics (such as other integration methods beyond our Euler integrator), and/or tests to further improve the library's completeness.
- **New demonstrations/tutorials:** If you would like to contribute to ngc-learn's tutorials/demonstrations, we happily welcome informative contributions that explore different ways of using the library, including how a particular classical or landmark model or system can be built, simulated, and/or analyzed with ngc-learn. Please adhere to the style of the other demonstrations (found under "Demonstrations" in the [documentation](https://ngc-learn.readthedocs.io/en/latest/#)) to ensure uniformity.

Beyond the above items, there are many other ways to help. In particular answering queries on the issue tracker, <!--and reviewing other developers' pull requests are very valuable contributions that decrease the burden on the project maintainers.-->
reporting issues that you are facing, and giving a "thumbs up" on issues that others report and that are relevant to you.

If your research or application uses ngc-learn, please send us an [email](mailto:ago@cs.rit.edu) and we will update our ["List of Papers/Publications" page](https://ngc-learn.readthedocs.io/en/latest/ngclearn_papers.html) to cite and include your work.
Furthermore, if you use ngc-learn in teaching tutorials, organizing workshops, or in the classroom, please let us know and we will link to your public resource(s) given that one of ngc-learn's goals is to make research and education in predictive processing and neurobiologically-motivated neural systems more accessible and open to individuals across disciplines.

Finally, it also helps us immensely if you spread the word: reference the project from your blog and articles, mention it in any of your slack/discord channels to interested groups, link to it from your website, or simply star it in GitHub.

Testing
-------
Before submitting your contributions, make sure that the changes do not break existing functionalities.
We have a handful of [tests](https://github.com/ago109/ngc-learn/tree/master/tests) for verifying the correctness of the code.
You can run all the tests with the following commands in the `tests/` folder of ngc-learn. Make sure that it does not throw any error before you proceed to the next step.
```sh
$ python test_identity.py
```

```sh
$ python test_gen_dynamics.py
```

```sh
$ python test_harmonium.py
```

Submission
----------
Please read the coding conventions below and make sure that your code is consistent with ours. When making a contribution, make a [pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests)
to ngc-learn with an itemized list of what you have done. When you submit a pull request, it is immensely helpful to include example script(s) that showcase the proposed changes and highlight any new APIs.
We always love to see more test coverage. When it is appropriate, add a new test to the [tests](https://github.com/ago109/ngc-learn/tree/master/tests) folder for checking the correctness of your code.

Coding Conventions
------------------
We value readability and adhere to the following coding conventions:
- Indent using four spaces (soft tabs)
- Always put spaces after list items and method parameters (e.g., `[1, 2, 3]` rather than `[1,2,3]`), and around operators and hash arrows (e.g., `x += 1` rather than `x+=1`)
- Use the [Google Python Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for the docstrings
- For scripts such as in [examples](https://github.com/ago109/ngc-learn/tree/master/examples) and [tests](https://github.com/ago109/ngc-learn/tree/master/tests), please include a docstring at the top of the file that describes the high-level purpose of the script and/or instructions on how to use the scripts (if relevant).

We look forward to your contributions. Thank you!

The Neural Adaptive Computing (NAC) Laboratory and the ngc-learn development team

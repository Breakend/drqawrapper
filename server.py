from flask import Flask, render_template, request, url_for, jsonify
from parlai.agents.drqa.drqa import DrqaAgent
from parlai.core.params import ParlaiParser
import logging
import torch
import random
import json
random.seed(42)

# Initialize the Flask application
app = Flask(__name__)


# Define a route for the action of the form, for example '/hello/'
# We are also defining which type of requests this route is
# accepting: POST requests in this case
@app.route('/ask', methods=['POST'])
def ask():
    json_dict = request.get_json(silent=True)

    article = json_dict['article']
    question = json_dict['question']
    #print(article)
    #print(question)
    global agent

    observation = {'text': '\n'.join([article, question]),
                   'episode_done': True}
    agent.observe(observation)
    reply = agent.act()
    return jsonify({'reply':reply})

# Run the app :)
if __name__ == '__main__':
    argparser = ParlaiParser(True, True)
    DrqaAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()

    # Set logging (only stderr)
    logger = logging.getLogger('DrQA')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Set cuda
    opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
    if opt['cuda']:
        logger.info('[ Using CUDA (GPU %d) ]' % opt['gpu'])
        torch.cuda.set_device(opt['gpu'])

    # Load document reader
    assert('pretrained_model' in opt)
    global agent
    agent = DrqaAgent(opt)

    app.run(
        host="0.0.0.0",
        port=int("8888")
    )

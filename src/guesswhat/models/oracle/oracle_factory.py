from guesswhat.models.oracle.oracle_film import FiLM_Oracle
from guesswhat.models.oracle.oracle_network import OracleNetwork


# stupid factory class to create networks

def create_oracle(config, no_words, reuse=False, device=''):

    network_type = config["type"]
    no_answers = 3

    if network_type == "film":
        return FiLM_Oracle(config, no_words, no_answers, device=device, reuse=reuse)
    elif network_type == "baseline":
        return OracleNetwork(config, no_words, no_answers, device=device, reuse=reuse)
    else:
        assert False, "Invalid network_type: should be: baseline/cbn"



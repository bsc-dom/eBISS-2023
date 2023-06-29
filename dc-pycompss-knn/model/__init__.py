import re
import os

from dataclay import Client


def slurm_nodelist_expander(nodelist_expression):
    for match in re.finditer(r'[0-9a-z]+(\[[0-9,-]+\])?', nodelist_expression):
        # We are only interested in the whole match
        token = match.group(0)

        # This is a bit redundant, given that we have a group in the regular expression
        # but at the end of the day we have to refine and split is an equally good tool
        if '[' not in token:
            yield token.strip()
        else:
            prefix, suffix = token.strip().split("[")
            suffix = suffix[:-1]  # strip the ending "]" character
            for subtoken in suffix.split(","):
                if "-" in subtoken:
                    # We have to expand a range
                    start, end = subtoken.split("-")
                    
                    # Note that the end will always be the longest string (think 3-12 or 3-144)
                    str_templating = "%%s%%0%dd" % len(end)
                    for i in range(int(start), int(end)+1):
                        yield str_templating % (prefix, i)
                else:
                    # No need to expand any range
                    yield f"{prefix}{subtoken}"


# Get the value of the SLURM_JOB_NODELIST environment variable
node_list = os.environ.get('SLURM_JOB_NODELIST')

if node_list is not None:
    print("SLURM_JOB_NODELIST contains: %s" % node_list)

    # Expand
    nodes = list(slurm_nodelist_expander(node_list))
    print("Parsed into: %s" % nodes)

    client = Client(host=nodes[0], username="bscuser", password="bscpassword", dataset="hpcdataset")
    client.start()

    print("dataClay client started")

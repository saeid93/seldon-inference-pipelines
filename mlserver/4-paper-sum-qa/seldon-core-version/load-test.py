from barazmoon import MLServerBarAzmoon

gateway_endpoint="localhost:32000"
deployment_name = 'sum-qa'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"


workload = [10, 7, 4, 12, 15]
data=["""
Après des décennies en tant que pratiquant d'arts martiaux et coureur, Wes a "trouvé" le yoga en 2010.
Il en est venu à apprécier que son ampleur et sa profondeur fournissent un merveilleux lest pour stabiliser
le corps et l'esprit dans le style de vie rapide et axé sur la technologie d'aujourd'hui ;
le yoga est un antidote au stress et une voie vers une meilleure compréhension de soi et des autres.
Il est instructeur de yoga certifié RYT 500 du programme YogaWorks et s'est formé avec des maîtres contemporains,
dont Mme Maty Ezraty, co-fondatrice de YogaWorks et maître instructeur des traditions Iyengar et Ashtanga,
ainsi qu'une spécialisation avec M. Bernie. Clark, un maître instructeur de la tradition Yin.
Ses cours reflètent ces traditions, où il combine la base fondamentale d'un alignement précis avec des éléments
d'équilibre et de concentration. Ceux-ci s'entremêlent pour aider à fournir une voie pour cultiver une conscience
de vous-même, des autres et du monde qui vous entoure, ainsi que pour créer un refuge contre le style de vie rapide
et axé sur la technologie d'aujourd'hui. Il enseigne à aider les autres à réaliser le même bénéfice de la pratique dont il a lui-même bénéficié.
Mieux encore, les cours de yoga sont tout simplement merveilleux :
ils sont à quelques instants des exigences de la vie où vous pouvez simplement prendre soin de vous physiquement et émotionnellement.
    """]
data_shape = [1]
http_method = 'post'
data_type = 'text'

load_tester = MLServerBarAzmoon(
    endpoint=endpoint,
    http_method=http_method,
    workload=workload,
    data=data,
    data_shape=data_shape,
    data_type=data_type)

load_tester.start()
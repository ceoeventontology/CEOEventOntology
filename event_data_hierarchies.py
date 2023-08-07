# Created by ceoeventontology at 2023/2/9
parent_son_type = {
    # 33 types, e.g., Life:Die
    'ace2005': {
        'life': ['be-born', 'marry', 'divorce', 'injure', 'die'],
        'movement': ['transport'],
        'transaction': ['transfer-ownership', 'transfer-money'],
        'business': ['start-org', 'merge-org', 'declare-bankruptcy', 'end-org'],
        'conflict': ['attack', 'demonstrate'],
        'contact': ['meet', 'phone-write'],
        'personnel': ['start-position', 'end-position', 'nominate', 'elect'],
        'justice': ['arrest-jail', 'release-parole', 'trial-hearing', 'charge-indict', 'sue', 'convict', 'sentence',
                    'fine', 'execute', 'extradite', 'acquit', 'appeal', 'pardon'],
    },
    # 168 types + 8 virtual types, e.g., Change_event_time (actual there are 176 in addition)
    'maven': {
        'Event': ['Sentiment', 'Scenario', 'Change', 'Possession', 'Action'],
        'Sentiment': ['Agree_or_refuse_to_act', 'Deciding', 'Change_sentiment', 'Suspicion', 'Coming_to_believe',
                      'Suasion', 'GiveUp', 'Helping', 'Protest', 'Rewards_and_punishments', 'Risk', 'Quarreling',
                      'Warning', 'Surrendering', 'Aiming', 'Request', 'Commitment', 'Labeling', 'Revenge'],
        'Agree_or_refuse_to_act': ['Ratification', 'Sign_agreement'],
        'Change_sentiment': ['Convincing'],
        'Helping': ['Assistance', 'Supporting', 'Collaboration', 'Rescuing'],
        'Rewards_and_punishments': ['Award'],
        'Request': ['Imposing_obligation'],
        'Scenario': ['Emergency', 'Incident', 'Rite', 'Catastrophe', 'Competition', 'Lighting', 'Confronting_problem',
                     'Resolve_problem', 'Process_end', 'Process_start', 'Achieve'],
        'Change': ['Influence', 'Being_in_operation', 'Openness', 'Forming_relationships', 'Becoming',
                   'Change_event_time', 'Cause_change_of_strength', 'Cause_to_be_included', 'Change_tool',
                   'Cause_to_make_progress', 'Cause_change_of_position_on_a_scale', 'AlterBadState', 'Change_of_leadership',
                   'Cause_to_amalgamate', 'Dispersal', 'Coming_to_be', 'GetReady', 'Reforming_a_system'],
        'Influence': ['Having_or_lacking_access', 'Causation', 'Preventing_or_letting', 'Control', 'Limiting'],
        'Having_or_lacking_access': ['Hindering'],
        'Control': ['Conquering'],
        'Cause_change_of_strength': ['Recovering'],
        'Recovering': ['Cure'],
        'Cause_to_be_included': ['Becoming_a_member', 'Participation'],
        'Becoming_a_member': ['Employment'],
        'Cause_change_of_position_on_a_scale': ['Expansion'],
        'AlterBadState': ['Bodily_harm', 'Damaging', 'Death'],
        'Damaging': ['Destroying'],
        'Coming_to_be': ['Presence'],
        'Possession': ['Getting', 'Giving', 'Sending', 'Bringing', 'Renting', 'Earnings_and_losses', 'Expensiveness',
                       'Carry_goods', 'Exchange', 'Cost'],
        'Getting': ['Receiving', 'Commerce_buy'],
        'Giving': ['Submitting_documents', 'Supply', 'Commerce_pay', 'Commerce_sell'],
        'Action': ['Hold', 'Practice', 'Using', 'CauseToBeHidden', 'Communication', 'Come_together', 'Name_conferral',
                   'Violence', 'Legality', 'Wearing', 'Institutionalization', 'Creating', 'Motion', 'Know', 'Spatial',
                   'Education_teaching', 'Choosing', 'Arranging', 'Preserving'],
        'Using': ['Use_firearm', 'Expend_resource'],
        'CauseToBeHidden': ['Removing', 'Hiding_objects'],
        'Communication': ['Telling', 'Expressing_publicly', 'Reporting', 'Adducing', 'Response'],
        'Expressing_publicly': ['Statement'],
        'Reporting': ['Reveal_secret'],
        'Come_together': ['Social_event'],
        'Violence': ['Surrounding', 'Attack', 'Military_operation', 'Terrorism', 'Bearing_arms', 'Defending', 'Killing'],
        'Surrounding': ['Besieging'],
        'Military_operation': ['Hostile_encounter'],
        'Legality': ['Justifying', 'Legal_rulings', 'Criminal_investigation', 'Committing_crime', 'Judgment_communication'],
        'Legal_rulings': ['Prison', 'Extradition', 'Releasing', 'Arrest'],
        'Committing_crime': ['Theft', 'Robbery', 'Kidnapping'],
        'Creating': ['Create_artwork', 'Manufacturing', 'Building', 'Recording'],
        'Create_artwork': ['Writing', 'Publishing'],
        'Motion': ['Motion_directional', 'Body_movement', 'Self_motion', 'Patrolling', 'Traveling'],
        'Body_movement': ['Ingestion', 'Breathing', 'Vocalizations'],
        'Self_motion': ['Escaping'],
        'Traveling': ['Arriving', 'Departing', 'Temporary_stay'],
        'Know': ['Perception_active', 'Check', 'Finding', 'Research', 'Scrutiny', 'Scouring', 'Testing'],
        'Spatial': ['Emptying', 'Filling', 'Placing', 'Connect', 'Containing'],
    },
    # # 139 event types, e.g., Type/Subtype/Sub-subtype
    # 'RAMS': {
    #     ''
    # }
}
RAMS = dict()
with open('./resources/event_datasets/LDCOntology.txt', 'r') as f:
    LDCOntology_info = f.readlines()
read_flag = False
AIDA_event_types = 0
for idx, line in enumerate(LDCOntology_info):
    if line.startswith('# Event types'):
        read_flag = True
    if line.startswith('# Event argument types'):
        break
    if read_flag:
        if line.startswith('ldcOnt:'):
            AIDA_event_types += 1
            type_list = line[len('ldcOnt:'):].split()[0].split('.')
            for pos, type in enumerate(type_list):
                if pos == 0:
                    continue
                if type == 'n/a':
                    break
                if type_list[pos-1] not in RAMS:
                    RAMS[type_list[pos-1]] = [type]
                elif type not in RAMS[type_list[pos - 1]]:
                    RAMS[type_list[pos - 1]].append(type)
# AIDA_event_types: 139 49
# print(f'AIDA_event_types: {AIDA_event_types}', len(RAMS))
parent_son_type['rams'] = RAMS

son_parent_type = dict()
for dataset, info in parent_son_type.items():
    son_parent_type[dataset] = dict()
    for parent, son_list in info.items():
        for son in son_list:
            if son not in son_parent_type[dataset]:
                son_parent_type[dataset][son] = [parent]
            else:
                son_parent_type[dataset][son].append(parent)


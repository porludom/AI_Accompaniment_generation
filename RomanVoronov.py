import mido
from typing import List
import random
import numpy as np

input = "input1.mid"
output = "RomanVoronovOutput1-"

major_scale = (0, 2, 4, 5, 7, 9, 11)  # 2 2 1 2 2 2 1
minor_scale = (0, 2, 3, 5, 7, 8, 10)  # 2 1 2 2 1 2 2
note_names = ("C", "CD", "D", "DE", "E", "F", "FG", "G", "GA", "A", "AB",
              "B",)  # tuple for identifying a notes by finding a remainder by dividing by 12 in order to remove octave dependance. We divide by 12 because we have 12 notes in one octave
scales = {}
offset_for_chords = 36

for number, note_name in enumerate(note_names):  # create scales for every key in both major and minor
    scales[note_name] = [(number + step) % 12 for step in major_scale]  # all major
    scales[note_name + "m"] = [(number + step) % 12 for step in minor_scale]  # all minor


class KeyIdentifier:
    melody_notes = None
    melody = None
    key_freq = {}

    def __init__(self) -> None:

        for number, note_name in enumerate(note_names):
            self.key_freq[note_name] = 0  # number of notes in melody that coincide with scale of given key
            self.key_freq[note_name + "m"] = 0  # number of notes in melody that coincide with scale of given key

    def setMelody(self, melody: mido.MidiTrack):

        self.melody = melody  # save original song
        self.melody_notes = [message.note % 12 for message in self.melody if message.type == "note_on"]  # got all notes
        for key in self.key_freq:  # set to 0 all coincides
            self.key_freq[key] = 0

    def determineKey(self):  # number of note in scale from 0 to 11

        for note in self.melody_notes:  # Firstly calculate two the most appropriate keys according to coincidence of notes to keys scales
            for key, value in scales.items():  # value is a list of notes in the scale this key
                if note in value:  # if current melody note in scale of this key then
                    self.key_freq[key] = self.key_freq[key] + 1  # +1 point to key if we have coincided note.
        total_max = 0
        rel_keys = []  # here we will have two relative keys
        for key, value in keyident.key_freq.items():  # find max value in dict
            if value > total_max:
                total_max = value

        for key in keyident.key_freq.keys():  # find keys with max value
            if keyident.key_freq[key] is total_max:
                rel_keys.append(key)

        last_note = (self.melody_notes[-1]) % 12

        # Check in this order: I -> V -> IV -> III
        for val in [0, 4, 3, 2]:
            for key in rel_keys:
                if scales[key][val] is last_note:
                    return key
        # next code called if last note says almost nothing
        first_note = (self.melody_notes[0]) % 12
        for val in [0, 4, 3, 2]:
            for key in rel_keys:
                if scales[key][val] is first_note:
                    return key


class ChordBuilder:
    major_steps = (0, 4, 7)
    minor_steps = (0, 3, 7)
    sus2_steps = (0, 2, 7)
    sus4_steps = (0, 5, 7)
    dim_steps = (0, 3, 6)

    def __init__(self, key: str) -> None:
        self.scale = scales[key]
        self.isMajor = True if not key.endswith("m") else False  # to properly create major/minor chords

    def createMajorChord(self, first_note) -> List[int]:
        return [first_note + step for step in ChordBuilder.major_steps]

    def createMinorChord(self, first_note) -> List[int]:
        return [first_note + step for step in ChordBuilder.minor_steps]

    def createSUS2Chord(self, first_note) -> List[int]:
        return [first_note + step for step in ChordBuilder.sus2_steps]

    def createSUS4Chord(self, first_note) -> List[int]:
        return [first_note + step for step in ChordBuilder.sus4_steps]

    def createDIMChord(self, first_note) -> List[int]:
        return [first_note + step for step in ChordBuilder.dim_steps]

    def getPool(self) -> List[List[int]]:
        """
        Returns a list of available chords in the key that given by scale in constructor
        """
        # Firstly create all minor and major chords
        pool = []

        triad_major = (self.createMajorChord, self.createMinorChord, self.createMinorChord, self.createMajorChord,
                       self.createMajorChord, self.createMinorChord, self.createDIMChord)

        triad_minor = (
            self.createMinorChord, self.createDIMChord, self.createMajorChord, self.createMinorChord,
            self.createMinorChord,
            self.createMajorChord, self.createMajorChord)

        # all chords are created according to tables from assignment description
        if self.isMajor:
            for idx, func in enumerate(triad_major):
                pool.append(func(self.scale[idx]))
            # sus2 and sus4 next
            for idx, note in enumerate(self.scale):
                if (idx != 2) and (idx != 6):
                    pool.append(self.createSUS2Chord(note))

            for idx, note in enumerate(self.scale):
                if (idx != 3) and (idx != 6):
                    pool.append(self.createSUS4Chord(note))
        else:
            for idx, func in enumerate(triad_minor):
                pool.append(func(self.scale[idx]))
            # sus2 and sus4 next
            for idx, note in enumerate(self.scale):
                if (idx != 1) and (idx != 4):
                    pool.append(self.createSUS2Chord(note))

            for idx, note in enumerate(self.scale):
                if (idx != 1) and (idx != 5):
                    pool.append(self.createSUS4Chord(note))

        return pool


class EvolutionaryAlgorithm:

    def __init__(self, melody: List[int],
                 elite: int, numOfIndividuals: int,
                 numOfGenes: int,
                 generationsNumber: int,
                 numOfMutations: int,
                 numOfMutationsInsideMutation: int,
                 numOfCrossovers: int,
                 numOfRandomGeneCrossovers: int) -> None:

        self.numOfGenerations = generationsNumber  # how many iterations we will have
        self.numOfMutations = numOfMutations  # how many individuals to mutate
        self.numOfCrossovers = numOfCrossovers  # how many offspring to get using crossover
        self.numOfRandomGeneCrossovers = numOfRandomGeneCrossovers  # how many random genes to crossover in one iteration
        self.numOfGenes = numOfGenes  # how many genes we have in population
        self.numOfMutationsInsideMutation = numOfMutationsInsideMutation  # how many chords to mutate
        self.numOfIndividuals = numOfIndividuals  # how many individuals we have at once.
        self.melody = melody
        self.elite = elite

    def get_random_chord(self) -> List[int]:  # one chord among 17 available
        """
        Returns random chord
        """
        return pool[random.randrange(0, 17)]

    def get_random_individual(self) -> List[List[int]]:  # one accompanement
        """
        Returns random individual that consist of given number of chords(genes)
        """
        return [self.get_random_chord() for i in range(self.numOfGenes)]

    def get_random_population(self) -> List[List[List[int]]]:  # a bunch of accompanements
        """
        Returns random population
        """
        return [self.get_random_individual() for i in range(self.numOfIndividuals)]

    def get_chord_fitness(self, chord: List[int], fragment: List[int]) -> int:
        # 10 point for every note intersection among melody part and chord
        fitness = 5
        for note in fragment:
            if note % 12 in chord:
                fitness = fitness + 10
        # penalty for sus chords
        if (chord[1] - chord[0] == 2) or (chord[2] - chord[0] == 2) or (chord[2] - chord[1] == 2):
            fitness -= 5

        if chord[2] + offset_for_chords > 50:
            fitness -= 4
        return fitness

    def get_individual_fitness(self, individual: List[List[int]]) -> int:
        """
        This function returns the sum of fitnesses of all genes of the given individual
        """
        return sum([self.get_chord_fitness(chord, fragments[idx]) for idx, chord in enumerate(individual)])

    def get_population_sum_fitness(self, population: List[List[List[int]]]):
        """
        This function returns the sum of fitnesses of all individuals in given population
        """
        return sum([self.get_individual_fitness(individual) for individual in population])

    def crossover(self, population: List[List[List[int]]]):
        """
        Do given by numOfCrossovers field number of crossovers
        Every time it swaps a random parts of chromosomes of two parents.
        1st part: all genes before some random gene
        2nd part: all genes after some random gene
        Uses roulette wheel to determine parents for offsprings
        Returns a new population that consist of some elite individuals and offsprings.
        :rtype: List[List[List[int]]]
        """

        fitness = [self.get_individual_fitness(individual) for individual in population]

        total_fit = sum(fitness)
        probs = [fitness_value / total_fit for fitness_value in fitness]

        new_population = []

        for i in range(self.numOfCrossovers):  # add offsprings
            gene = random.randrange(1, self.numOfGenes)

            first = population[np.random.choice(len(population), p=probs)]  # individual
            second = population[np.random.choice(len(population), p=probs)]  # individual

            offspring = first[:gene]  # individual
            offspring[gene:] = second[gene:]
            new_population.append(offspring)
        for individual in population[-self.elite:]:  # add elite
            new_population.append(individual)

        return new_population

    def mutation(self, population: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        Do mutation of a given population.
        Mutation means some genes(chords in this case) are replaced by other random genes(chords)
        """
        for i in range(self.numOfMutations):
            idx = random.randrange(0, self.numOfIndividuals)  # which individual to mutate
            for k in range(self.numOfMutationsInsideMutation):
                randomGene = random.randrange(0, self.numOfGenes)  # choose random gene

                population[idx][randomGene] = pool[random.randrange(0, 17)]  # mutate this random gene
        return population

    def selection(self, population: List[List[List[int]]]) -> List[List[List[int]]]:
        population_fitness = [self.get_individual_fitness(individual) for individual in population]

        indices_population_sorted = np.argsort(population_fitness)  # by increasing of fitness

        sorted_population = []

        for index in indices_population_sorted:  # sort population to easily select elite
            sorted_population.append(population[index])
        sorted_population = self.crossover(sorted_population)  # do crossover

        return self.mutation(sorted_population)  # do mutation

    def evolution(self):

        population = self.get_random_population()
        maxval = 0  # here I have a maximum fitness that ever obtained
        freq = 0  # how many times in a row we obtain the maximum fitness above
        for i in range(self.numOfGenerations):

            population = self.selection(population)  # new population
            fitnesses = [self.get_individual_fitness(individual) for individual in population]
            max_fit = max(fitnesses)
            if max_fit == maxval:
                freq += 1
                if freq == 100:  # we have a maximum. there are no local maximums, therefore we found global and can stop
                    break
            elif max_fit > maxval:
                maxval = max_fit
                freq = 0

        fitnesses = [self.get_individual_fitness(individual) for individual in population]
        max_fit = max(fitnesses)
        for idx, individual in enumerate(population):
            if fitnesses[idx] == max_fit:
                return individual  # return one of the best accompaniments we have


track = mido.MidiFile(input)  # load a midi
keyident = KeyIdentifier()  # utility object
keyident.setMelody(track.tracks[1])
key = keyident.determineKey()  # identify the key
output = output + str(key) + ".mid"
chordBuilder = ChordBuilder(key)  # utility object based on key
pool = chordBuilder.getPool()  # we have only 17 available chord without taking into account inversed triads

accord_length = track.ticks_per_beat

fragments = []  # list of lists of notes. every list of notes belong to one chord
# fragments are used to calculate fitness of chord. One fragment belongs to one chord

time = 0
fragment = []  # list of notes
for message in track.tracks[1]:
    if message.type == "note_on":
        time = time + message.time
        if message.time != 0:  # means pause
            while time >= accord_length:
                fragments.append(fragment)
                fragment = []
                time = time - accord_length
        fragment.append(message.note % 12)

    elif message.type == "note_off":
        time = time + message.time
        while time >= accord_length:
            fragments.append(fragment)
            fragment = []
            time = time - accord_length
            if time != 0:  # one note for two fragments
                fragment.append(message.note % 12)
if fragment:
    fragments.append(fragment)

number_of_chords = fragments.__len__()  # equal to number of genes

accompaniment = EvolutionaryAlgorithm(melody=keyident.melody_notes,
                                      numOfGenes=number_of_chords,
                                      numOfMutations=50,
                                      numOfIndividuals=500,
                                      numOfMutationsInsideMutation=1,
                                      generationsNumber=1000,
                                      elite=350,
                                      numOfCrossovers=150,
                                      numOfRandomGeneCrossovers=int(number_of_chords / 2)).evolution()

# now include accompanement in music

trackWithChords = mido.MidiTrack()

trackWithChords.append(mido.MetaMessage(type="track_name", name="accompaniment", time=0))
trackWithChords.append(mido.Message(type="program_change", channel=0, program=0, time=0))

for chord in accompaniment:
    for note in chord:
        trackWithChords.append(
            mido.Message(type="note_on", channel=0, note=note + offset_for_chords, velocity=40, time=0))
    trackWithChords.append(
        mido.Message(type="note_off", channel=0, note=chord[0] + offset_for_chords, velocity=40, time=accord_length))
    trackWithChords.append(
        mido.Message(type="note_off", channel=0, note=chord[1] + offset_for_chords, velocity=40, time=0))
    trackWithChords.append(
        mido.Message(type="note_off", channel=0, note=chord[2] + offset_for_chords, velocity=40, time=0))

trackWithChords.append(mido.MetaMessage(type="end_of_track", time=0))
track.tracks.append(trackWithChords)
track.save(filename=output)
print("Done!")

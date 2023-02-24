# DL_musical_structure
Final Project for CPS452/552 'Deep Learning Theory and Applications'

This repositort holds the final project for the aforementioned class. The aim of the project is to perform automated musical analysis, given a MIDI file as input. A Convolutional NN, as well as a LSTM and a Transformer have been developed to perform this task. Our models outputs an ordered, sequence of phrases. Consider the follownig output example:

i4A4B8A4A4b4B8A4A4b4b4A4A4b4A4o3

In this phrase, each letter is indicative of a phrase, with the accompanying number the amount of beats this phrase lasts for. Recurring letters imply the re-emergence of the phrase. Lower case letters indicate phrases without melody lines - for example, b4 refers to the same phrase as B4, but without the melody line (they could share the same chord sequence and percussive elements). The letters I and O refer to intro and outro respectively - in the structure output above, we have a 4 beat intro (without a melody), and a 3 beat outro (also without a melody). The letter X refers to any unique phrase that only appears once in the song - each x phrase is distinct (otherwise they would be categorized as a phrase by another letter).

This project was a shared effort between Nhi Nguyen, Simone Genetin, Sam Kouteili, and Michael Zhou. 


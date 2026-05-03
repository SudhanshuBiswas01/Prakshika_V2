# Prompt Diff: Permissive vs Strict

Comparison of responses using permissive vs strict system prompts.
Model: `llama-3.1-8b-instant` | temperature=0

---

## Query 1: What is Newton's law of gravitation and what is the formula?

### Permissive
```
Newton's law of gravitation is a fundamental concept in physics that describes the force of gravity between two objects. According to Newton's law of gravitation, every point mass attracts every other point mass by a force acting along the line intersecting both points.

The formula for Newton's law of gravitation is:

F = G \* (m1 \* m2) / r^2

Where:

- F is the gravitational force between the two objects
- G is the gravitational constant (6.67 x 10^-11 N m^2 kg^-2)
- m1 and m2 are the masses of the two objects
- r is the distance between the centers of the two objects

This formula shows that the gravitational force between two objects depends on the product of their masses and the inverse square of the distance between them.
```

### Strict
```
Newton's law of gravitation states that every point mass attracts every other point mass by a force acting along the line intersecting both points. The force is proportional to the product of their masses and inversely proportional to the square of the distance between their centers. The formula is:

F = G \* (m1 \* m2) / r^2

where F is the gravitational force, G is the gravitational constant (6.67 x 10^-11 Nm^2/kg^2), m1 and m2 are the masses of the two objects, and r is the distance between their centers.

[Source: ch1_p11_0002]
```

---

## Query 2: Explain Mendeleev's periodic table and its limitations.

### Permissive
```
Mendeleev's periodic table is a tabular arrangement of elements, organized based on their chemical and physical properties. It was developed by Dmitri Mendeleev in the late 19th century. The table consists of vertical columns called groups and horizontal rows called periods. The elements are arranged in a way that elements with similar properties are placed in the same group, and elements with similar atomic masses are placed in the same period.

Mendeleev's periodic table was a significant contribution to the field of chemistry, as it helped to identify patterns and relationships between elements. The table was based on the idea that the properties of elements are a periodic function of their atomic masses. Mendeleev predicted the existence of undiscovered elements and left gaps in the table for them, which were later filled when the elements were discovered.

However, Mendeleev's periodic table had some limitations. Some of the limitations include:

1. **Ambiguity in the position of hydrogen**: Hydrogen shows similarity with both halogens (group VII) and alkali metals (group I), making it difficult to decide its correct position in the table.
2. **Isotopes**: The discovery of isotopes, which have the same chemical properties but different atomic masses, posed a challenge in placing them in the table.
3. **Non-uniform rise in atomic mass**: The rise in atomic mass does not appear to be uniform, making it difficult to predict how many elements could be discovered between two heavy elements.
4. **Cobalt and nickel**: The whole number atomic mass of cobalt (Co) and nickel (Ni) is the same, leading to ambiguity regarding their sequence in the table.
5. **Lack of understanding of atomic structure**: At the time Mendeleev developed the table, the scientific community did not have a clear understanding of the atomic structure, which limited the accuracy of the table.

Despite these limitations, Mendeleev's periodic table was a significant step forward in the development of chemistry and laid the foundation for the modern periodic table, which is based on atomic numbers rather than atomic masses.
```

### Strict
```
Mendeleev's periodic table was organized on the basis of the chemical and physical properties of the elements, including molecular formulae of hydrides and oxides, melting points, boiling points, and densities of the elements and their hydrides and oxides [Source: ch2_p28_0064].

The vertical columns in Mendeleev's periodic table are called groups, while the horizontal rows are called periods [Source: ch2_p28_0064].

Mendeleev's periodic table demonstrates the following merits: it is progressive, allows for revision of old inferences, and shows freedom in science to revise the old inference by using more advanced means and methods of doing experiments [Source: ch2_p29_0065].

However, Mendeleev's periodic table has some limitations, including:

1. Vacant places in the table, where atomic masses were predicted [Source: ch2_p29_0065].
2. Ambiguity in the names of some elements, indicated by a question mark before the symbol [Source: ch2_p29_0065].
3. The whole number atomic mass of elements cobalt (Co) and nickel (Ni) is the same, causing ambiguity regarding their sequence [Source: ch2_p30_0069].
4. Isotopes were discovered long after Mendeleev put forth the periodic table, posing a challenge in placing them in the table [Source: ch2_p30_0069].
5. The rise in atomic mass does not appear to be uniform, making it difficult to predict how many elements could be discovered between two heavy elements [Source: ch2_p30_0069].
6. The position of hydrogen is ambiguous, showing similarity with both halogens and alkali metals [Source: ch2_p30_0069].

These limitations were addressed by the development of the modern periodic table, which is based on atomic numbers rather than atomic masses [Source: ch2_p30_0070].
```

---

## Query 3: What is the boiling point of water at different altitudes? **(OUT OF SCOPE)**

### Permissive
```
Unfortunately, the provided context does not cover the boiling point of water at different altitudes. However, I can provide a general answer based on my knowledge.

The boiling point of water decreases with an increase in altitude due to the decrease in atmospheric pressure. At sea level, the boiling point of water is 100°C (212°F) at standard atmospheric pressure. However, at higher altitudes, the boiling point of water decreases because the atmospheric pressure is lower.

Here's a rough estimate of the boiling point of water at different altitudes:

- At 1,000 meters (3,300 feet) above sea level, the boiling point of water is approximately 94°C (201°F).
- At 2,000 meters (6,600 feet) above sea level, the boiling point of water is approximately 90°C (194°F).
- At 3,000 meters (9,800 feet) above sea level, the boiling point of water is approximately 86°C (187°F).
- At 4,000 meters (13,100 feet) above sea level, the boiling point of water is approximately 82°C (180°F).

Please note that these values are approximate and can vary depending on the specific location and atmospheric conditions.
```

### Strict
```
I don't have that in my study materials.
```

> **Observation**: The permissive prompt likely answers using general knowledge. The strict prompt should refuse.

---


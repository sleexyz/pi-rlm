/**
 * Memorable run name generator (adjective-noun pairs).
 */

const ADJECTIVES = [
	"amber", "ancient", "bold", "bright", "calm",
	"coral", "crisp", "dark", "deep", "dusty",
	"fading", "fern", "fierce", "foggy", "frozen",
	"gentle", "gilded", "golden", "hollow", "iron",
	"jade", "keen", "lush", "misty", "mossy",
	"pale", "quiet", "rapid", "rustic", "silver",
	"sleek", "solar", "stark", "steep", "stellar",
	"stone", "swift", "tidal", "vast", "wild",
];

const NOUNS = [
	"basin", "birch", "brook", "canyon", "cedar",
	"cliff", "cove", "creek", "dawn", "delta",
	"dune", "ember", "fern", "field", "fjord",
	"frost", "glade", "grove", "haven", "heath",
	"lake", "ledge", "marsh", "meadow", "mesa",
	"moss", "oak", "peak", "pine", "pond",
	"reef", "ridge", "river", "shore", "spruce",
	"stone", "vale", "wren", "summit", "thorn",
];

export function generateRunName(): string {
	const adj = ADJECTIVES[Math.floor(Math.random() * ADJECTIVES.length)];
	const noun = NOUNS[Math.floor(Math.random() * NOUNS.length)];
	return `${adj}-${noun}`;
}

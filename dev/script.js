const translateButton = document.getElementById('translateButton');
const inputText = document.getElementById('inputText');
const outputText = document.getElementById('outputText');

translateButton.addEventListener('click', async () => {
    const inputSentence = inputText.value.trim();
    
    if (inputSentence === '') {
        alert('Please enter a sentence to translate.');
        return;
    }

    try {
        const response = await fetch('http://localhost:8000/translate', { // Update the URL here
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ input_text: inputSentence })
        });

        if (response.ok) {
            const data = await response.json();
            outputText.textContent = `Translation: ${data.translated_sentence}`;
        } else {
            alert('An error occurred while translating.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while translating.');
    }
});

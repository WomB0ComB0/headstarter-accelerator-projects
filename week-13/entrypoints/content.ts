/**
 * Content script for providing AI text completion functionality in text input fields
 * 
 * This content script adds real-time AI text completion suggestions to text inputs and textareas
 * across all web pages. It displays suggestions in a floating element that follows the input,
 * and allows users to accept suggestions using the Tab key.
 */

interface CompletionMessage {
  type: 'getCompletion';
  prompt: string;
}

interface TestMessage {
  type: 'test';
}

type Message = CompletionMessage | TestMessage;

export default defineContentScript({
  // Match pattern to run this content script on all URLs
  matches: ['<all_urls>'],
  
  /**
   * Main content script function that sets up the text completion functionality
   */
  main() {
    console.log('[Content Script] Initializing text completion functionality');

    console.log('[Content Script] Script loaded');
    console.log('[Content Script] Current URL:', window.location.href);

    // Track the currently focused input element
    let activeInput: HTMLInputElement | HTMLTextAreaElement | null = null;
    // Element used to display completion suggestions
    let completionSpan: HTMLSpanElement | null = null;

    /**
     * Creates and styles the floating element used to display completion suggestions
     * @returns {HTMLSpanElement} Styled span element for showing completions
     */
    function createCompletionElement(): HTMLSpanElement {
      console.log('[Content Script] Creating completion element');
      const span = document.createElement('span');
      span.style.position = 'absolute';
      span.style.color = '#888';
      span.style.pointerEvents = 'none';
      span.style.userSelect = 'none';
      return span;
    }

    /**
     * Positions the completion element to match the input's position and styling
     * @param {HTMLElement} input - The input element to match
     * @param {string} text - The completion text to display
     */
    function positionCompletionElement(input: HTMLElement, text: string): void {
      if (!completionSpan) return;
      
      console.log('[Content Script] Positioning completion element');
      const rect = input.getBoundingClientRect();
      const style = window.getComputedStyle(input);
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      
      // Match position and styling of input element
      completionSpan.style.top = `${rect.top + scrollTop}px`;
      completionSpan.style.left = `${rect.left}px`;
      completionSpan.style.fontFamily = style.fontFamily;
      completionSpan.style.fontSize = style.fontSize;
      completionSpan.style.lineHeight = style.lineHeight;
      completionSpan.style.padding = style.padding;
      completionSpan.style.border = style.border;
      completionSpan.innerText = text;
    }

    /**
     * Requests a text completion from the background script
     * @param {string} prompt - The current input text to generate a completion for
     * @returns {Promise<string>} The generated completion text
     */
    async function getCompletion(prompt: string): Promise<string> {
      console.log('[Content Script] Requesting completion for prompt:', prompt);
      const completion = await browser.runtime.sendMessage<CompletionMessage, string>({ 
        type: 'getCompletion', 
        prompt 
      });
      console.log('[Content Script] Received completion:', completion);
      return completion;
    }

    /**
     * Handles input events by requesting and displaying completions
     * @param {Event} event - The input event
     */
    async function handleInput(event: Event): Promise<void> {
      const target = event.target as HTMLInputElement | HTMLTextAreaElement;
      if (!['INPUT', 'TEXTAREA'].includes(target.tagName)) return;

      console.log('[Content Script] Input detected:', {
        element: target.tagName,
        value: target.value
      });

      try {
        activeInput = target;
        const text = target.value;
        
        // Only request completion if there's text
        if (!text.trim()) return;

        const completion = await getCompletion(text);
        
        console.log('[Content Script] Received completion:', completion);

        if (!completionSpan) {
          completionSpan = createCompletionElement();
          document.body.appendChild(completionSpan);
        }
        
        positionCompletionElement(target, completion || '');
      } catch (error) {
        console.error('[Content Script] Error in handleInput:', error);
      }
    }

    /**
     * Handles keydown events to allow accepting completions with Tab
     * @param {KeyboardEvent} event - The keydown event
     */
    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === 'Tab' && activeInput && completionSpan) {
        console.log('[Content Script] Tab pressed - accepting completion');
        event.preventDefault();
        activeInput.value += completionSpan.innerText;
        completionSpan.innerText = '';
      }
    }

    /**
     * Clears completion when text selection changes
     */
    function handleSelectionChange(): void {
      if (completionSpan) {
        console.log('[Content Script] Selection changed - clearing completion');
        completionSpan.innerText = '';
      }
    }

    /**
     * Clears completion and active input tracking when focus is lost
     */
    function handleBlur(): void {
      if (completionSpan) {
        console.log('[Content Script] Focus lost - clearing completion');
        completionSpan.innerText = '';
      }
      activeInput = null;
    }

    // Set up event listeners with passive option
    console.log('[Content Script] Setting up event listeners');
    document.addEventListener('input', handleInput);
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('scroll', () => {
      if (activeInput && completionSpan) {
        console.log('[Content Script] Scroll detected - updating completion position');
        positionCompletionElement(activeInput, completionSpan.innerText);
      }
    }, { passive: true });
    document.addEventListener('selectionchange', handleSelectionChange, { passive: true });
    document.addEventListener('blur', handleBlur, true);

    // Add a visible element to confirm script injection
    const debugElement = document.createElement('div');
    debugElement.style.position = 'fixed';
    debugElement.style.bottom = '10px';
    debugElement.style.right = '10px';
    debugElement.style.background = 'red';
    debugElement.style.padding = '5px';
    debugElement.style.zIndex = '9999';
    debugElement.textContent = 'Content Script Active';
    document.body.appendChild(debugElement);

    // Test message passing
    browser.runtime.sendMessage<TestMessage>({ type: 'test' }).then(response => {
      console.log('Background response:', response);
      debugElement.textContent += ' - Connected';
    }).catch(error => {
      console.error('Connection error:', error);
      debugElement.textContent += ' - Error';
    });
  }
});

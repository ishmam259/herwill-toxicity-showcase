// Hand-written placeholder posts (not from the competition data, so they are safe to publish).
// Obidit's data/examples.json replaces these when it lands (TASKS.md, Obidit's section).

export interface Example {
  text: string
  script: 'Bangla' | 'Banglish' | 'English'
}

export const EXAMPLES: Example[] = [
  { script: 'Bangla', text: 'মেয়েদের এত কথা বলা মানায় না, চুপ থাকাই ভালো।' },
  { script: 'Bangla', text: 'অনুষ্ঠানটা খুব সুন্দর হয়েছে, সবাইকে ধন্যবাদ।' },
  { script: 'Bangla', text: 'তুই একটা অসভ্য, এখান থেকে বের হয়ে যা।' },
  { script: 'Banglish', text: 'meye manush hoye eto boro boro kotha?' },
  { script: 'Banglish', text: 'khub valo laglo video ta, aro banan' },
  { script: 'English', text: 'Not bad for a girl, I guess.' },
  { script: 'English', text: 'Congratulations on the new job!' },
  { script: 'English', text: 'You are a disgusting idiot, nobody wants you here.' },
]

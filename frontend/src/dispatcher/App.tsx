import { useEffect, useRef, useState } from 'react';
import { MotionConfig, motion } from 'motion/react';
import { amount, number, routes, scenario, time, type DataStatus, type Filter, type Route } from './model';
import Icon from './Icon';
import Numeric from './Numeric';
import Forecast from './Forecast';
import ThemeSwitcher from './ThemeSwitcher';
import './styles/base.css';
import './styles/composition.css';
import '../styles/design-tokens.css';
import '../styles/fonts.css';
import './styles/dispatcher.css';
import './styles/theme.css';
const topics: Record<string, [
    string,
    string
]> = {
    source: ['О данных', 'Это визуальный демо-режим на условных данных. ML-сервис и онлайн-LLM пока не подключены. Шесть маршрутов иллюстрируют сценарии работы с набором из 1000 маршрутов. Дата данных — 01.11.2025, часовой пояс не указан.'],
    scenario: ['Параметры подачи', 'Вместимость и остаток у ворот — параметры сценария, которых нет в конкурсном датасете. Сначала определяется готовность груза, затем время вызова с учётом подачи. Объём к прибытию учитывает весь приход от среза до прибытия. Единицы — исходные счётчики, не тонны или м³.'],
    forecast: ['Как читать прогноз', '«За час» — объём за час, заканчивающийся в выбранной точке. Соседние окна перекрываются. Накопленный приход считается отдельно по получасовому потоку. Диапазон здесь иллюстративный.'],
    dispatch: ['Как определяется вызов', 'Вызов назначается за время подачи до готовности груза. Светлая полоса — время подачи, фиолетовая метка — вызов, круглая точка — готовность. Выберите маршрут кликом по строке.'],
    assistant: ['Об ассистенте', 'Это демонстрация размещения ассистента и ограниченных сценариев без внешней LLM: объяснение рекомендации, подача 90 минут, вызовы за час. В рабочем продукте агент и пульт будут читать один snapshot.'],
    quality: ['Качество прогноза', 'Метрика конкурса — WAPE + |Relative Bias|. Локальные исторические оценки и результат платформы показываются отдельно. Для условных данных прототипа метрика не вычисляется.'],
};
function Dispatcher() {
    const [selected, setSelected] = useState(471), [lead, setLead] = useState(60), [filter, setFilter] = useState<Filter>('all');
    const [query, setQuery] = useState(''), [metric, setMetric] = useState<'hourly' | 'inflow'>('hourly'), [point, setPoint] = useState(3);
    const [status, setStatus] = useState<DataStatus>('ready'), [question, setQuestion] = useState(''), [draft, setDraft] = useState('');
    const [topic, setTopic] = useState('source');
    const infoDialog = useRef<HTMLDialogElement>(null);
    const route = routes.find(r => r.id === selected)!;
    const stale = (r: Route) => status === 'partial' && r.id === 862;
    const plan = (r: Route, minutes = lead) => scenario(r, minutes, stale(r));
    const list = routes.filter(r => (!query || String(r.id).includes(query)) && (filter === 'all' || plan(r).kind === filter));
    const select = (id: number) => { setSelected(id); setQuestion(''); setDraft(''); };
    const info = (key: string) => { setTopic(key); infoDialog.current?.showModal(); };
    useEffect(() => {
        const pointer = () => { document.documentElement.dataset.input = 'pointer'; };
        const key = (event: KeyboardEvent) => {
            if (event.key === 'Tab' || event.key.startsWith('Arrow') || event.key === 'Home' || event.key === 'End')
                document.documentElement.dataset.input = 'keyboard';
        };
        document.addEventListener('pointerdown', pointer, true);
        document.addEventListener('keydown', key);
        return () => { document.removeEventListener('pointerdown', pointer, true); document.removeEventListener('keydown', key); };
    }, []);
    const infoButton = (key: string, label: string) => <button className="icon-btn info-button" onClick={() => info(key)} data-info={key} aria-label={label} title={label}><Icon name="info"/></button>;
    const filters = () => <section className="workspace-toolbar" aria-label="Поиск и фильтры"><label className="search"><Icon name="search"/><input id="route-search" type="search" placeholder="Номер маршрута" aria-label="Номер маршрута" value={query} onChange={e => setQuery(e.target.value)}/></label><div className="filter-tabs" role="group" aria-label="Срочность">{([['all', 'Все'], ['now', 'Сейчас'], ['soon', 'За час'], ['later', 'Позже']] as const).map(([id, label]) => <button key={id} data-filter={id} aria-pressed={filter === id} onClick={() => setFilter(id)}>{label}<span>{routes.filter(r => id === 'all' || plan(r).kind === id).length}</span></button>)}</div></section>;
    const empty = <div className="empty"><Icon name="search"/><h3>Маршруты не найдены</h3><button onClick={() => { setQuery(''); setFilter('all'); }}>Сбросить фильтры</button></div>;
    const p = plan(route);
    const calculation = <section className="calculation"><div className="block-heading"><h3>Готовность к подаче</h3></div><div className="calculation-settings"><div className="lead-control"><span>Подача</span><div className="segments" role="group" aria-label="Время подачи">{[30, 60, 90].map(n => <button key={n} data-lead={n} aria-pressed={lead === n} onClick={() => setLead(n)}>{n}<span> мин</span></button>)}</div>{infoButton('scenario', 'Параметры подачи')}</div></div>{p.total !== null && p.incoming !== null ? <><div className="decision-values"><div><span>У ворот</span><strong><Numeric value={route.backlog}/></strong></div><b>+</b><div><span>Придёт к {time(p.arrival!)}</span><strong><Numeric value={p.incoming}/></strong></div><b>=</b><div className="result"><span>К прибытию</span><strong><Numeric value={p.total}/></strong></div></div><div className="fill-track" aria-label={`Заполнение ${number(p.total / route.capacity * 100)} процентов`}><i style={{ width: Math.min(100, route.backlog / route.capacity * 100) + '%' }}/><span style={{ width: Math.max(0, Math.min(100 - route.backlog / route.capacity * 100, p.incoming / route.capacity * 100)) + '%' }}/></div><div className="fill-caption"><span><Numeric value={p.total / route.capacity * 100} format="percent"/> вместимости</span><span><Numeric value={route.capacity}/> ед.</span></div></> : <div className="unavailable">Расчёт появится после обновления наблюдений.</div>}<div className="decision-facts"><span><Icon name="clock"/>Прибытие <b>{p.arrival === null ? '—' : time(p.arrival)}</b></span><span><Icon name="truck"/>В пути <b>0 машин</b></span></div></section>;
    const forecast = <Forecast route={route} metric={metric} point={point} onMetric={setMetric} onPoint={setPoint} unavailable={stale(route)} info={() => info('forecast')}/>;
    const badge = <span className={`status ${p.kind}`}><Icon name={p.kind === 'now' ? 'truck' : 'clock'}/>{p.kind === 'stale' ? 'Нет свежих данных' : p.call === 0 ? 'Вызвать сейчас' : p.call === null ? 'Не требуется' : 'Вызвать в ' + p.label}</span>;
    const detail = <><div className="detail-header"><h2>{route.id}</h2>{badge}</div>{forecast}{calculation}</>;
    const table = list.length ? <div className="table-wrap" tabIndex={0} aria-label="Маршруты в очереди"><table><thead><tr><th>Маршрут</th><th>Вызов</th><th>У ворот, тыс.</th><th className="extra-column">К прибытию, тыс.</th></tr></thead><tbody>{list.map(r => <tr key={r.id} className={`route-row ${r.id === selected ? 'selected' : ''}`} data-select-route={r.id} aria-selected={r.id === selected} onClick={() => select(r.id)}><td><button className="route-button" data-route={r.id} aria-label={`Выбрать маршрут ${r.id}`} aria-pressed={r.id === selected}>{r.id}</button></td><td><span className={`call-label ${plan(r).kind}`}>{plan(r).label}</span></td><td><Numeric value={r.backlog} format="number"/></td><td className="extra-column">{plan(r).total === null ? '—' : <Numeric value={plan(r).total!} format="number"/>}</td></tr>)}</tbody></table><div className="table-count">{list.length} маршрутов</div></div> : empty;
    const reply = () => {
        if (stale(route))
            return <p>Нет свежих наблюдений. Обновите снимок, чтобы получить рекомендацию.</p>;
        if (/90|подач|измен|сценар/.test(question.toLowerCase())) {
            const alt = plan(route, 90);
            return <><h3>Если подача — 90 минут</h3><p>Вызов: <strong>{alt.label}</strong>. Прибытие: <strong>{alt.arrival === null ? '—' : time(alt.arrival)}</strong>.</p><p>Ожидается <strong>{amount(alt.total!)} ед.</strong> — {number(alt.total! / route.capacity * 100)}% вместимости.</p><button className="assistant-action" onClick={() => setLead(90)}>Применить 90 мин</button></>;
        }
        if (/час|очеред|кого/.test(question.toLowerCase()))
            return <><h3>Вызовы в ближайший час</h3><div className="assistant-route-links">{routes.filter(r => plan(r).call !== null && plan(r).call! <= 60).map(r => <button key={r.id} data-route={r.id} onClick={() => select(r.id)}><strong>{r.id}</strong><span>{plan(r).label}</span></button>)}</div></>;
        if (question && !/почему|решени|объяс|пояс|выз/.test(question.toLowerCase()))
            return <p>В прототипе можно разобрать рекомендацию, сравнить подачу 90 минут или посмотреть вызовы за час.</p>;
        return <><h3>{p.call === 0 ? 'Вызвать сейчас' : 'Вызов в ' + p.label}</h3><p>До прибытия в {time(p.arrival!)} к готовому грузу добавится <strong>{amount(p.incoming!)} ед.</strong></p><dl className="assistant-evidence"><div><dt>К прибытию</dt><dd>{amount(p.total!)}</dd></div><div><dt>Вместимость</dt><dd>{amount(route.capacity)}</dd></div><div><dt>Заполнение</dt><dd>{number(p.total! / route.capacity * 100)}%</dd></div></dl><button className="text-button" onClick={() => info('scenario')}>Основание расчёта</button></>;
    };
    const assistantPanel = <motion.aside key="assistant" id="assistant-drawer" className="panel assistant-drawer" aria-labelledby="assistant-title" initial={false}><header className="assistant-drawer-header"><h2 id="assistant-title">Ассистент</h2><div>{infoButton('assistant', 'Об ассистенте')}</div></header><div id="assistant-content"><section className="assistant"><div className="assistant-context"><strong>Маршрут {selected}</strong><span>Срез 10:30</span></div><div className="assistant-thread" aria-live="polite">{question && question !== 'Почему?' && <p className="assistant-user">{question}</p>}<div className="assistant-reply">{reply()}</div></div><div className="assistant-prompts">{['Почему?', 'Подача 90 мин', 'Вызовы за час'].map(text => <button key={text} data-prompt={text} onClick={() => setQuestion(text)}>{text}</button>)}</div><form className="assistant-composer" onSubmit={e => { e.preventDefault(); if (draft.trim()) {
        setQuestion(draft.trim());
        setDraft('');
    } }}><input id="assistant-input" name="question" aria-label="Вопрос ассистенту" placeholder="Спросить о маршруте…" value={draft} onChange={e => setDraft(e.target.value)}/><button type="submit" aria-label="Отправить вопрос"><Icon name="send"/></button></form></section></div></motion.aside>;
    return <><a className="skip" href="#app">К рабочему экрану</a><header className="app-header"><div className="app-brand"><span className="rwb-mark">RWB</span><i className="brand-divider"/><span className="app-name">Поток</span></div><div className="header-actions"><ThemeSwitcher/><span className="snapshot"><i />01.11 · 10:30</span><button className="icon-btn" onClick={() => info('quality')} aria-label="Качество прогноза"><Icon name="chart"/></button><button className="icon-btn" data-info="source" onClick={() => info('source')} aria-label="О данных и прототипе"><Icon name="info"/></button></div></header>
  <main className="page-scroll" tabIndex={-1} aria-label="Рабочий экран"><div id="app" className="layout-desk">
  {status === 'error' && <div className="notice" role="alert"><Icon name="info"/><span>Снимок не обновлён. Показаны последние доступные данные.</span><button onClick={() => setStatus('ready')}>Повторить</button></div>}
  {status === 'partial' && <div className="notice" role="status"><Icon name="info"/><span>Маршрут 862: нет свежих наблюдений.</span><button onClick={() => select(862)}>Открыть</button></div>}
  {status === 'loading' ? <section className="loading panel" aria-busy="true"><div className="loading-label">Обновляем прогноз<button onClick={() => setStatus('ready')}>Показать данные</button></div><div className="skeleton short"/><div className="skeleton plot"/>{[0, 1, 2].map(i => <div key={i} className="skeleton row"/>)}</section> :
            <motion.div initial={false} animate={{ opacity: 1 }} transition={{ duration: .18 }} className="desk-grid">
  <section className="panel queue"><div className="block-heading"><h2>Очередь <span className="queue-total">{routes.length}</span></h2>{infoButton('dispatch', 'Как формируется очередь')}</div>{filters()}{table}</section><section className="panel route-detail">{detail}</section>{assistantPanel}
  </motion.div>}</div></main>

  <dialog ref={infoDialog} id="info-dialog" aria-labelledby="dialog-title" onClick={e => { if (e.target === e.currentTarget) {
        const r = e.currentTarget.getBoundingClientRect();
        if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)
            e.currentTarget.close();
    } }}><div className="dialog-header"><h2 id="dialog-title">{topics[topic][0]}</h2><button className="icon-btn" aria-label="Закрыть" onClick={() => infoDialog.current?.close()}><Icon name="close"/></button></div><div id="dialog-content"><p>{topics[topic][1]}</p>{topic === 'source' && <><div className="info-section"><h3>Проверить состояние</h3><div className="state-buttons">{([['ready', 'Данные'], ['loading', 'Загрузка'], ['error', 'Ошибка'], ['partial', 'Частично']] as const).map(([id, label]) => <button key={id} data-state={id} aria-pressed={status === id} onClick={() => { setStatus(id); infoDialog.current?.close(); }}>{label}</button>)}</div></div><div className="info-section"><h3>Выбранный экран</h3><p>Очередь, выбранный маршрут и ассистент работают в общей сетке.</p></div></>}</div></dialog>
  <div id="announcement" className="sr-only" aria-live="polite">Маршрут {selected}</div></>;
}
export default function App() {
    return <MotionConfig reducedMotion="user"><Dispatcher /></MotionConfig>;
}

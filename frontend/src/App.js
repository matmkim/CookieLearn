import './App.css';
import { useState , useEffect, useRef} from 'react';
/*eslint-disable */

function App() {
  const [input,setInput] = useState("");
  const [history,setHistory] = useState("봇: 어떻게 도와드릴까요?\n\n");
  const categories = ['정치','경제', '사회', '세계','IT/과학'];
  const [hotIssues, setHotIssues] = useState(['총선','쌍특검법 거부권','태영 건설','금리 인하']);
  const chatHistoryRef = useRef(null);
  const helpRef = useRef(1);

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [history]);

  const inputChange = (e) => {
    setInput(e.target.value);
  };

  const sendMessage = () => {
    setHistory((prevState)=>{
      return prevState+"사용자: "+input+"\n\n";
    });
    fetchJSON(input);
  };

  const categorySearch = (category) => {
    setHistory((prevState)=>{
      return prevState+"사용자: 최근 "+category+" 분야 뉴스 알려줘.\n\n";
    });
    fetchJSON("최근 "+category+" 분야 뉴스 알려줘.");
  };

  const issueSearch = (issue) => {
    setHistory((prevState)=>{
      return prevState+"사용자: "+issue+ " 관련 대표적인 기사 알려줘.\n\n";
    });
    fetchJSON(issue+"관련 대표적인 기사 알려줘");
  };

  const fetchJSON = (input) => {
    console.log(input);
    fetch('/api/chat', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({ 'user_input': input })
    })
    .then(response => response.json())
    .then(data => {
      setHistory((prevState)=>{
        return prevState+"봇: "+data.bot_output+"\n\n";
      });
      console.log(data);
    });
    setInput("");
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="title">📰 News' Balance</h1>
        <nav className="categories">
          {categories.map((category,idx)=>{
            return <a onClick={()=>{categorySearch(category)}}>
              {category}
            </a>
          })}
        </nav>
      </header>
      <main className="body">
        <aside className="sideBar">
          <h2>Hot Issues</h2>
          <hr/>
          <ul>
            {hotIssues.map((hotIssue,idx)=>{
              return <li>
                <a onClick={()=>{issueSearch(hotIssue)}}>
                  {hotIssue}
                </a>
            </li>
            })}
          </ul>
          
          <div ref={helpRef} className="help-modal">
            <div className="modal-content">
              <p>[주제 선택 기능 / Hot Issue 선택 기능]
              -  주제 / Hot issue를 선택하면 대화형으로 채팅으로 이어져 server의 답변을 받게 된다.
              </p>
              <p>[Free Chat 기능]
              - 학습된 모델 하에서 뉴스에 관한 Chat이 자유롭게 가능하다
              </p>
              <p>[명령어 List 기능]
              -  Bot의 기능을 더욱 효율적으로 사용할 수 있도록 가능 명령어 제시
              </p>
              <button className="close-button" onClick={()=>{helpRef.current.style.display='none'}}>Close</button>
            </div>
          </div>
          <button onClick={()=>{helpRef.current.style.display='flex'}}> Help  🙋 </button>
        </aside>
        <section className="main">
          <h2> 채팅 창</h2>
          <div className="chat-window">
            <div ref={chatHistoryRef} style = {{whiteSpace: 'pre-line',}}>{history}</div>
            <div className="chatting-box">
              <input placeholder="메세지를 입력하세요..." value ={input} onChange={inputChange}/>
              <button className="send-button" onClick={sendMessage}>보내기</button>
            </div>
          </div>
          <div className = "commands">
            <h2>가능한 명령어</h2>
            <div className="command-list">
              <div>
                <p>관련 기사 (유사 bias score)</p>
                <button onClick={()=>{setInput(input+" 유사 기사 알려줘")}}>유사 뉴스</button>
              </div>
              <div>
                <p>관련 기사 (반대 bias score)</p>
                <button onClick={()=>{setInput(input+" 상반되는 기사 알려줘")}}>반대 뉴스</button>
              </div>
              <div>
                <p>Bias score 계산</p>
                <button onClick={()=>{setInput(input+"의 편향 정도를 알려줘")}}> Bias Score</button>
              </div>
              <div>
                <p>중립 요약</p>
                <button onClick={()=>{setInput(input+" 중립 기사로 요약해줘")}}>중립 요약</button>
              </div>
              <div>
                <p>채팅 창 초기화</p>
                <button onClick={()=>{setHistory("봇: 어떻게 도와드릴까요?\n\n")}}>초기화</button>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
